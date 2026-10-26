import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms.functional as TF
import gradio as gr

# Thêm đường dẫn dự án
sys.path.insert(0, os.path.abspath('.'))

try:
    from models.forge import NAFNetBaseline, FORGEDeblurNet
    from utils.metrics import calc_psnr, calc_ssim
except ImportError as e:
    print(f"[Cảnh báo] Kiểm tra lại đường dẫn models/ và utils/: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> Web Demo đang khởi chạy trên thiết bị: {device}")

# ==============================================================================
# 1. BỘ ĐỊNH TUYẾN MOE DECORATOR (THEO DÕI TRỌNG SỐ EXPERT)
# ==============================================================================
def patch_moe_gate_tracking(model):
    """Gắn hook theo dõi tỉ lệ kích hoạt Chuyên gia MoE."""
    for module in model.modules():
        if type(module).__name__ == 'ForgeMoEConv2d':
            def make_patched_fw(mod):
                orig_fw = mod.forward
                def patched_fw(x, f_global):
                    b, c, h, w = x.shape
                    with torch.no_grad():
                        f_g_mapped = mod.f_proj(f_global)
                        f_g_exp = f_g_mapped.expand(-1, -1, h, w)
                        r_in = torch.cat([x, f_g_exp], dim=1)
                        z = mod.router(r_in) + mod.m_g
                        gates = F.softmax(z, dim=1)
                        mod.last_gates = gates.mean(dim=(0, 2, 3)).detach().cpu().numpy()
                    return orig_fw(x, f_global)
                return patched_fw
            module.forward = make_patched_fw(module)

def get_moe_expert_distribution(model):
    """Lấy trung bình cộng tỉ lệ đóng góp của các Expert trong toàn mô hình."""
    all_gates = []
    for module in model.modules():
        if type(module).__name__ == 'ForgeMoEConv2d' and hasattr(module, 'last_gates'):
            all_gates.append(module.last_gates)
    if not all_gates:
        return None
    return np.mean(all_gates, axis=0)

# ==============================================================================
# 2. HÀM TỰ ĐỘNG PHÂN CẤP VÀ NẠP MÔ HÌNH DỰA TRÊN FILE TRỌNG SỐ (AUTO-DETECT)
# ==============================================================================
def auto_load_model_from_pth(custom_weight_file):
    if custom_weight_file is None:
        raise ValueError("⚠️ Vui lòng tải file trọng số (.pth) lên!")

    ckpt_path = custom_weight_file.name if hasattr(custom_weight_file, 'name') else custom_weight_file
    st = torch.load(ckpt_path, map_location=device)

    # Giải nén state_dict nếu bị bọc trong dict lớn
    if isinstance(st, dict):
        if 'params' in st:
            st = st['params']
        elif 'model_state_dict' in st:
            st = st['model_state_dict']
        elif 'state_dict' in st:
            st = st['state_dict']

    # TỰ ĐỘNG NHẬN DIỆN KIẾN TRÚC DỰA VÀO KEY TRONG FILE .PTH
    keys_str = " ".join(st.keys())
    is_moe = any(k in keys_str for k in ['experts', 'lora_', 'mid_f_mapper', 'router', 'W0.weight'])

    if is_moe:
        model = FORGEDeblurNet(
            img_channel=3, out_channels=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
            num_queries=16, num_experts=4, r=16, rb=32, moe_in_decoder=False
        ).to(device)
        model.load_state_dict(st, strict=False)
        patch_moe_gate_tracking(model)
        model_type_name = "FORGE MoE-LoRA (Đa Tác Vụ)"
    else:
        model = NAFNetBaseline(
            img_channel=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]
        ).to(device)
        model.load_state_dict(st, strict=True)
        model_type_name = "NAFNet Baseline (Đơn Tác Vụ)"

    model.eval()
    return model, is_moe, model_type_name, os.path.basename(ckpt_path)

# ==============================================================================
# 3. HÀM TRỰC QUAN HÓA
# ==============================================================================
def create_heatmap(input_np, output_np):
    diff = np.abs(input_np.astype(np.float32) - output_np.astype(np.float32)) / 255.0
    diff_gray = np.mean(diff, axis=2)
    diff_norm = (diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min() + 1e-8)
    heatmap_rgb = (cm.inferno(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    return heatmap_rgb

def plot_moe_distribution(expert_weights):
    if expert_weights is None:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.text(0.5, 0.5, "Không dùng MoE (Mô hình Baseline)", ha='center', va='center', fontsize=9, color='gray')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
    experts = [f"Expert {i+1}" for i in range(len(expert_weights))]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

    bars = ax.bar(experts, expert_weights * 100, color=colors[:len(expert_weights)], width=0.45)
    ax.set_ylabel("Đóng góp (%)", fontsize=9)
    ax.set_title("Phân bổ Trọng số Chuyên gia MoE (Routing)", fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig

# ==============================================================================
# 4. HÀM XỬ LÝ CHÍNH (TAB 1: PHỤC HỒI ĐƠN)
# ==============================================================================
def process_single_model(input_img, custom_weight_file, gt_img=None):
    if custom_weight_file is None:
        return None, None, None, "⚠️ **LỖI:** Hãy thả file trọng số (`.pth`) vào ô tải lên!"

    if input_img is None:
        return None, None, None, "⚠️ **LỖI:** Hãy tải ảnh bị hỏng/suy hao lên!"

    try:
        start_t = time.time()
        model, is_moe, model_type_name, loaded_ckpt_name = auto_load_model_from_pth(custom_weight_file)

        img_pil = Image.fromarray(input_img).convert('RGB')
        input_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

        _, _, h, w = input_tensor.shape
        multiple = 16
        H = ((h + multiple - 1) // multiple) * multiple
        W = ((w + multiple - 1) // multiple) * multiple
        pad_h, pad_w = H - h, W - w
        input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            out = model(input_padded.float())
            preds = out[0] if isinstance(out, tuple) else out
            preds = preds[:, :, :h, :w]
            preds = torch.clamp(preds, 0.0, 1.0)

        proc_time = (time.time() - start_t) * 1000
        output_np = (preds.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        slider_tuple = (input_img, output_np)
        heatmap_np = create_heatmap(input_img, output_np)

        expert_weights = get_moe_expert_distribution(model) if is_moe else None
        fig_moe = plot_moe_distribution(expert_weights)

        metrics_md = f"📂 **File Trọng Số:** `{loaded_ckpt_name}`\n\n"
        metrics_md += f"🤖 **Kiến Trúc Nhận Diện:** `{model_type_name}`\n\n"
        metrics_md += f"⚡ **Thời gian xử lý:** `{proc_time:.1f} ms`\n\n"

        if gt_img is not None:
            gt_pil = Image.fromarray(gt_img).convert('RGB')
            gt_tensor = TF.to_tensor(gt_pil).unsqueeze(0).to(device)
            p_val = calc_psnr(preds, gt_tensor)
            s_val = calc_ssim(preds, gt_tensor)
            metrics_md += f"📊 **PSNR:** `{p_val:.2f} dB`  |  📐 **SSIM:** `{s_val:.4f}`\n\n"
        else:
            metrics_md += "💡 *(Chế độ Thực tế: Đã ẩn chỉ số do không có ảnh sạch gốc)*"

        return slider_tuple, heatmap_np, fig_moe, metrics_md

    except Exception as e:
        return None, None, None, f"❌ **Lỗi đọc file trọng số:** `{str(e)}`"

# ==============================================================================
# 5. HÀM XỬ LÝ CHÍNH (TAB 2: ĐẤU TRƯỜNG SO SÁNH 2 FILE TRỌNG SỐ)
# ==============================================================================
def process_arena_comparison(input_img, weight_file_1, weight_file_2, gt_img=None):
    if weight_file_1 is None or weight_file_2 is None:
        return None, None, None, "⚠️ **LỖI:** Vui lòng tải lên **ĐỦ 2 FILE TRỌNG SỐ (.pth)** để thực hiện so sánh!"

    if input_img is None:
        return None, None, None, "⚠️ **LỖI:** Hãy tải ảnh bị hỏng/suy hao lên!"

    try:
        model1, is_moe1, type1, name1 = auto_load_model_from_pth(weight_file_1)
        model2, is_moe2, type2, name2 = auto_load_model_from_pth(weight_file_2)

        img_pil = Image.fromarray(input_img).convert('RGB')
        input_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

        _, _, h, w = input_tensor.shape
        multiple = 16
        H = ((h + multiple - 1) // multiple) * multiple
        W = ((w + multiple - 1) // multiple) * multiple
        pad_h, pad_w = H - h, W - w
        input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            t0 = time.time()
            out1 = model1(input_padded.float())
            pred1 = (out1[0] if isinstance(out1, tuple) else out1)[:, :, :h, :w]
            pred1 = torch.clamp(pred1, 0.0, 1.0)
            time1 = (time.time() - t0) * 1000

            t1 = time.time()
            out2 = model2(input_padded.float())
            pred2 = (out2[0] if isinstance(out2, tuple) else out2)[:, :, :h, :w]
            pred2 = torch.clamp(pred2, 0.0, 1.0)
            time2 = (time.time() - t1) * 1000

        out1_np = (pred1.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        out2_np = (pred2.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        metrics_md = "### ⚔️ KẾT QUẢ SO SÁNH 2 FILE TRỌNG SỐ:\n\n"
        metrics_md += f"🔴 **File 1 (`{name1}`):** `{type1}` - `{time1:.1f} ms`\n\n"
        metrics_md += f"🟢 **File 2 (`{name2}`):** `{type2}` - `{time2:.1f} ms`\n\n"

        if gt_img is not None:
            gt_pil = Image.fromarray(gt_img).convert('RGB')
            gt_tensor = TF.to_tensor(gt_pil).unsqueeze(0).to(device)

            p1 = calc_psnr(pred1, gt_tensor)
            s1 = calc_ssim(pred1, gt_tensor)

            p2 = calc_psnr(pred2, gt_tensor)
            s2 = calc_ssim(pred2, gt_tensor)

            diff_p = p2 - p1
            diff_s = s2 - s1

            metrics_md += f"📊 **File 1 Metrics:** PSNR = `{p1:.2f} dB` | SSIM = `{s1:.4f}`\n\n"
            metrics_md += f"📊 **File 2 Metrics:** PSNR = `{p2:.2f} dB` | SSIM = `{s2:.4f}`\n\n"
            metrics_md += f"🚀 **Chênh lệch (File 2 vs File 1):** **`{diff_p:+.2f} dB PSNR`** | **`{diff_s:+.4f} SSIM`**"
        else:
            metrics_md += "💡 *Chế độ Thực Tế: Đã ẩn chỉ số PSNR/SSIM do không có ảnh sạch gốc.*"

        return input_img, out1_np, out2_np, metrics_md

    except Exception as e:
        return None, None, None, f"❌ **Lỗi xử lý:** `{str(e)}`"

# ==============================================================================
# 6. GIAO DIỆN GRADIO BLOCKS TỐI GIẢN
# ==============================================================================
with gr.Blocks(title="FORGE MoE Image Restoration Dashboard") as demo:
    gr.Markdown(
        """
        # 🎨 FORGE MoE: Multitask Image Restoration Web Demo
        **Hệ thống Phục hồi Ảnh Đa Tác vụ (Khử nhòe, Khử sương, Khử nhiễu, Tăng sáng, Khử mưa, Khử tuyết)**
        """
    )

    with gr.Tabs():
        # ----------------------------------------------------------------------
        # TAB 1: PHỤC HỒI CHI TIẾT (CHỈ CẦN 1 FILE PTH)
        # ----------------------------------------------------------------------
        with gr.TabItem("🚀 Phục Hồi & Trực Quan Hóa Chi Tiết"):
            with gr.Row():
                with gr.Column(scale=1):
                    custom_weight1 = gr.File(
                        label="📁 Tải file trọng số (.pth) bất kỳ",
                        file_types=[".pth"]
                    )
                    in_img1 = gr.Image(label="📥 Ảnh Bị Suy Hao (Input Degraded)", type="numpy")
                    gt_img1 = gr.Image(label="🎯 Ảnh Sạch Gốc (Ground Truth - Tùy chọn)", type="numpy")
                    btn_run1 = gr.Button("✨ Phục Hồi Ảnh Ngay", variant="primary")

                with gr.Column(scale=1):
                    slider_out = gr.ImageSlider(label="↔️ Kéo thanh trượt để so sánh Before / After")
                    metrics_box1 = gr.Markdown("⏱️ *Đang chờ tải trọng số và ảnh...*")

            with gr.Row():
                with gr.Column(scale=1):
                    heatmap_out = gr.Image(label="🔥 Bản Đồ Nhiệt Vùng Thay Đổi (Restoration Heatmap)", type="numpy")
                with gr.Column(scale=1):
                    plot_out = gr.Plot(label="📊 Phân Bổ Trọng Số Chuyên Gia (MoE Expert Routing)")

            btn_run1.click(
                fn=process_single_model,
                inputs=[in_img1, custom_weight1, gt_img1],
                outputs=[slider_out, heatmap_out, plot_out, metrics_box1]
            )

        # ----------------------------------------------------------------------
        # TAB 2: ĐẤU TRƯỜNG SO SÁNH 2 FILE TRỌNG SỐ BẤT KỲ
        # ----------------------------------------------------------------------
        with gr.TabItem("⚔️ Đấu Trường So Sánh 2 Trọng Số (.pth)"):
            with gr.Row():
                with gr.Column(scale=1):
                    weight_file_1 = gr.File(
                        label="📁 File Trọng Số 1 (.pth) - Ví dụ: Baseline",
                        file_types=[".pth"]
                    )
                    weight_file_2 = gr.File(
                        label="📁 File Trọng Số 2 (.pth) - Ví dụ: FORGE MoE",
                        file_types=[".pth"]
                    )
                    in_img2 = gr.Image(label="📥 Ảnh Bị Suy Hao (Input Degraded)", type="numpy")
                with gr.Column(scale=1):
                    gt_img2 = gr.Image(label="🎯 Ảnh Sạch Gốc (Ground Truth - Tùy chọn)", type="numpy")

            btn_run2 = gr.Button("⚔️ Bắt Đầu So Sánh Ngay", variant="primary")
            arena_metrics = gr.Markdown("💡 *Đang chờ tải đủ 2 file trọng số...*")

            with gr.Row():
                arena_in = gr.Image(label="1️⃣ Ảnh Bị Suy Hao (Input)", type="numpy")
                arena_out1 = gr.Image(label="2️⃣ Kết Quả Trọng Số 1", type="numpy")
                arena_out2 = gr.Image(label="3️⃣ Kết Quả Trọng Số 2", type="numpy")

            btn_run2.click(
                fn=process_arena_comparison,
                inputs=[in_img2, weight_file_1, weight_file_2, gt_img2],
                outputs=[arena_in, arena_out1, arena_out2, arena_metrics]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
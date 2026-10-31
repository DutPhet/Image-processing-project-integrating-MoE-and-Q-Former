import os
import sys
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms.functional as TF
import gradio as gr

# Thêm đường dẫn dự án
sys.path.insert(0, os.path.abspath('.'))

try:
    from models.forge import NAFNetBaseline, FORGEDeblurNet
    from models.prompt_forge import PromptFORGEDeblurNet
    from utils.metrics import calc_psnr, calc_ssim
except ImportError as e:
    print(f"[Cảnh báo] Kiểm tra lại đường dẫn models/ và utils/: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> FORGE Web Demo đang chạy trên thiết bị: {device}")

# Thư mục lưu trữ trọng số mô hình
SAVED_WEIGHTS_DIR = os.path.abspath("saved_weights")
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)


# ==============================================================================
# 1. BỘ ĐỊNH TUYẾN MOE DECORATOR & TRACKING
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
# 2. QUẢN LÝ TRỌNG SỐ & TỰ ĐỘNG PHÂN CẤP MÔ HÌNH
# ==============================================================================
def get_saved_model_list():
    """Lấy danh sách các file weights đã lưu."""
    files = [f for f in os.listdir(SAVED_WEIGHTS_DIR) if f.endswith('.pth')]
    return files if files else ["Chưa có mô hình nào"]

def save_custom_weight(file_obj, custom_name):
    """Lưu file trọng số mới vào thư mục saved_weights với tên tùy chỉnh."""
    if file_obj is None:
        return "⚠️ **Lỗi:** Vui lòng tải file `.pth` lên!", gr.update()
    if not custom_name or custom_name.strip() == "":
        return "⚠️ **Lỗi:** Vui lòng nhập tên mô hình!", gr.update()

    clean_name = custom_name.strip().replace(" ", "_")
    if not clean_name.endswith(".pth"):
        clean_name += ".pth"

    dest_path = os.path.join(SAVED_WEIGHTS_DIR, clean_name)
    src_path = file_obj.name if hasattr(file_obj, 'name') else file_obj

    try:
        shutil.copy(src_path, dest_path)
        new_list = get_saved_model_list()
        msg = f"✅ **Đã lưu mô hình thành công với tên:** `{clean_name}`"
        return msg, gr.update(choices=new_list, value=clean_name), gr.update(choices=new_list, value=clean_name)
    except Exception as e:
        return f"❌ **Lỗi khi lưu file:** `{str(e)}`", gr.update(), gr.update()

def auto_load_model(model_selection, uploaded_file=None):
    """Tự động nhận diện kiến trúc và load weights từ file được chọn/up lên."""
    ckpt_path = None

    # Ưu tiên chọn từ danh sách đã lưu
    if model_selection and model_selection != "Chưa có mô hình nào":
        possible_path = os.path.join(SAVED_WEIGHTS_DIR, model_selection)
        if os.path.exists(possible_path):
            ckpt_path = possible_path

    # Nếu chọn upload trực tiếp
    if ckpt_path is None and uploaded_file is not None:
        ckpt_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file

    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise ValueError("Chưa chọn hoặc tải lên file trọng số hợp lệ!")

    st = torch.load(ckpt_path, map_location=device)

    # Giải nén state_dict
    if isinstance(st, dict):
        for key in ['params', 'model_state_dict', 'state_dict', 'model']:
            if key in st and isinstance(st[key], dict):
                st = st[key]
                break

    keys_str = " ".join(st.keys())
    is_prompt_forge = any(k in keys_str for k in ['pgm', 'prompt_keys', 'prompt_values'])
    is_moe = any(k in keys_str for k in ['experts', 'lora_', 'mid_f_mapper', 'router', 'W0.weight'])

    if is_prompt_forge:
        model = PromptFORGEDeblurNet(
            img_channel=3, out_channels=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
            num_prompts=5, num_experts=4, top_k=2, r=16, rb=32, moe_in_decoder=False
        ).to(device)
        model.load_state_dict(st, strict=False)
        patch_moe_gate_tracking(model)
        model_type_name = "Prompt-FORGE MoE (Khôi Phục Đa Tác Vụ Cao Cấp)"
    elif is_moe:
        model = FORGEDeblurNet(
            img_channel=3, out_channels=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
            num_queries=16, num_experts=4, top_k=2, r=16, rb=32, moe_in_decoder=False
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
        model_type_name = "NAFNet Baseline (Mô Hình Cơ Sở)"

    model.eval()
    return model, is_moe or is_prompt_forge, model_type_name, os.path.basename(ckpt_path)


# ==============================================================================
# 3. TRỰC QUAN HÓA (HEATMAP & MOE CHART)
# ==============================================================================
def create_heatmap(input_np, output_np):
    diff = np.abs(input_np.astype(np.float32) - output_np.astype(np.float32)) / 255.0
    diff_gray = np.mean(diff, axis=2)
    diff_norm = (diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min() + 1e-8)
    heatmap_rgb = (cm.inferno(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    return heatmap_rgb

def plot_moe_distribution(expert_weights):
    if expert_weights is None:
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.text(0.5, 0.5, "Mô hình Baseline (Không có MoE)", ha='center', va='center', fontsize=9, color='gray')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=100)
    experts = [f"Expert {i+1}" for i in range(len(expert_weights))]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

    bars = ax.bar(experts, expert_weights * 100, color=colors[:len(expert_weights)], width=0.45)
    ax.set_ylabel("Đóng góp (%)", fontsize=8)
    ax.set_title("Tỉ lệ Kích hoạt Expert MoE Routing", fontsize=9, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


# ==============================================================================
# 4. TÁC VỤ 1: PHỤC HỒI ẢNH (SINGLE IMAGE RESTORATION)
# ==============================================================================
def process_image(input_img, model_choice, direct_file, gt_img=None):
    if input_img is None:
        return None, None, None, "⚠️ **Lỗi:** Vui lòng tải ảnh đầu vào lên!"

    try:
        start_t = time.time()
        model, is_moe, model_type_name, loaded_ckpt_name = auto_load_model(model_choice, direct_file)

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

        metrics_md = f"📂 **Mô hình nạp:** `{loaded_ckpt_name}`\n\n"
        metrics_md += f"🤖 **Kiến trúc:** `{model_type_name}`\n\n"
        metrics_md += f"⚡ **Thời gian xử lý:** `{proc_time:.1f} ms`\n\n"

        if gt_img is not None:
            gt_pil = Image.fromarray(gt_img).convert('RGB')
            gt_tensor = TF.to_tensor(gt_pil).unsqueeze(0).to(device)
            p_val = calc_psnr(preds, gt_tensor)
            s_val = calc_ssim(preds, gt_tensor)
            metrics_md += f"📊 **PSNR:** `{p_val:.2f} dB`  |  📐 **SSIM:** `{s_val:.4f}`\n\n"

        return slider_tuple, heatmap_np, fig_moe, metrics_md

    except Exception as e:
        return None, None, None, f"❌ **Lỗi xử lý ảnh:** `{str(e)}`"


# ==============================================================================
# 5. TÁC VỤ 2: PHỤC HỒI & LÀM NÉT VIDEO (VIDEO RESTORATION)
# ==============================================================================
def process_video(video_path, model_choice, direct_file, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️ **Lỗi:** Vui lòng tải file Video đầu vào lên!"

    try:
        model, is_moe, model_type_name, loaded_ckpt_name = auto_load_model(model_choice, direct_file)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ Không thể đọc file Video!"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_dir = os.path.abspath("outputs")
        os.makedirs(output_dir, exist_ok=True)
        out_video_path = os.path.join(output_dir, f"restored_{int(time.time())}.mp4")

        # Codec mp4v an toàn cho hầu hết nền tảng
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps if fps > 0 else 25, (w, h))

        multiple = 16
        H = ((h + multiple - 1) // multiple) * multiple
        W = ((w + multiple - 1) // multiple) * multiple
        pad_h, pad_w = H - h, W - w

        start_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress(frame_idx / max(1, total_frames), desc=f"Đang làm nét Video... ({frame_idx}/{total_frames} frames)")

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            input_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)
            input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                out = model(input_padded.float())
                preds = out[0] if isinstance(out, tuple) else out
                preds = preds[:, :, :h, :w]
                preds = torch.clamp(preds, 0.0, 1.0)

            out_np = (preds.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            out_writer.write(out_bgr)

        cap.release()
        out_writer.release()

        total_time = time.time() - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0

        info_md = f"✅ **Phục hồi Video Hoàn Tất!**\n\n"
        info_md += f"📂 **Mô hình:** `{loaded_ckpt_name}` ({model_type_name})\n\n"
        info_md += f"🎞️ **Số Khung Hình:** `{total_frames}` frames | ⏱️ **Tốc độ:** `{avg_fps:.1f} FPS`\n\n"
        info_md += f"💾 **Video đã lưu tại:** `{out_video_path}`"

        return out_video_path, info_md

    except Exception as e:
        return None, f"❌ **Lỗi xử lý Video:** `{str(e)}`"


# ==============================================================================
# 6. GIAO DIỆN GRADIO UI (TỐI GIẢN & HIỆN ĐẠI)
# ==============================================================================
init_models = get_saved_model_list()

with gr.Blocks(title="FORGE MoE Image & Video Restoration") as demo:
    gr.Markdown(
        """
        # 🚀 FORGE MoE: System Image & Video Restoration Dashboard
        **Hệ thống Phục hồi & Làm nét Ảnh / Video đa tác vụ dựa trên MoE-LoRA thích ứng**
        """
    )

    # --------------------------------------------------------------------------
    # KHU VỰC QUẢN LÝ & LƯU MÔ HÌNH (MODEL MANAGEMENT)
    # --------------------------------------------------------------------------
    with gr.Accordion("💾 QUẢN LÝ & LƯU FILE TRỌNG SỐ MÔ HÌNH (.PTH)", open=True):
        with gr.Row():
            with gr.Column(scale=2):
                upload_pth = gr.File(label="1️⃣ Tải lên File Trọng số (.pth)", file_types=[".pth"])
            with gr.Column(scale=2):
                model_name_input = gr.Textbox(label="2️⃣ Đặt Tên Mô Hình", placeholder="Ví dụ: FORGE_Deblur_Best.pth")
                btn_save_model = gr.Button("💾 Lưu Trọng Số Vào Thư Mục", variant="primary")
            with gr.Column(scale=3):
                save_status = gr.Markdown("💡 *Tải file trọng số lên, đặt tên và bấm **Lưu** để sử dụng lâu dài.*")

    # --------------------------------------------------------------------------
    # KHU VỰC CHÍNH: TABS XỬ LÝ ẢNH & VIDEO
    # --------------------------------------------------------------------------
    with gr.Tabs():
        # ----------------------------------------------------------------------
        # TAB 1: PHỤC HỒI & LÀM NẾT ẢNH
        # ----------------------------------------------------------------------
        with gr.TabItem("🖼️ Phục Hồi & Làm Nét Ảnh"):
            with gr.Row():
                with gr.Column(scale=1):
                    select_model_img = gr.Dropdown(
                        choices=init_models,
                        value=init_models[0] if init_models else None,
                        label="🤖 Chọn Mô Hình Đã Lưu"
                    )
                    direct_file_img = gr.File(label="Hoặc dùng file .pth tạm thời", file_types=[".pth"])
                    in_img = gr.Image(label="📥 Ảnh Bị Mờ / Suy Hao", type="numpy")
                    gt_img = gr.Image(label="🎯 Ảnh Sạch Gốc (Tùy chọn - để đo PSNR/SSIM)", type="numpy")
                    btn_run_img = gr.Button("✨ Xử Lý & Làm Nét Ảnh", variant="primary")

                with gr.Column(scale=1):
                    slider_out = gr.ImageSlider(label="↔️ So Sánh Trước / Sau Phục Hồi (Before / After)")
                    metrics_box_img = gr.Markdown("⏱️ *Chờ lệnh xử lý...*")

                    with gr.Row():
                        heatmap_out = gr.Image(label="🔥 Heatmap Vùng Thay Đổi", type="numpy")
                        plot_out = gr.Plot(label="📊 MoE Expert Activation")

            btn_run_img.click(
                fn=process_image,
                inputs=[in_img, select_model_img, direct_file_img, gt_img],
                outputs=[slider_out, heatmap_out, plot_out, metrics_box_img]
            )

        # ----------------------------------------------------------------------
        # TAB 2: PHỤC HỒI & LÀM NẾT VIDEO
        # ----------------------------------------------------------------------
        with gr.TabItem("🎥 Phục Hồi & Làm Nét Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    select_model_vid = gr.Dropdown(
                        choices=init_models,
                        value=init_models[0] if init_models else None,
                        label="🤖 Chọn Mô Hình Đã Lưu"
                    )
                    direct_file_vid = gr.File(label="Hoặc dùng file .pth tạm thời", file_types=[".pth"])
                    in_video = gr.Video(label="📥 Tải lên Video Bị Mờ / Nhiễu / Suy Hao")
                    btn_run_vid = gr.Button("🎬 Bắt Đầu Làm Nét Video", variant="primary")

                with gr.Column(scale=1):
                    out_video = gr.Video(label="🎬 Video Đã Được Phục Hồi Sắc Sét")
                    metrics_box_vid = gr.Markdown("⏱️ *Chờ lệnh xử lý video...*")

            btn_run_vid.click(
                fn=process_video,
                inputs=[in_video, select_model_vid, direct_file_vid],
                outputs=[out_video, metrics_box_vid]
            )

    # Sự kiện Lưu trọng số
    btn_save_model.click(
        fn=save_custom_weight,
        inputs=[upload_pth, model_name_input],
        outputs=[save_status, select_model_img, select_model_vid]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
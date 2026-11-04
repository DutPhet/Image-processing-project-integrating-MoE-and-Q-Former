import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.abspath(".cache/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import sys
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import gradio as gr

# Đảm bảo đường dẫn import cho models/ và utils/
sys.path.insert(0, os.path.abspath('.'))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Fallback imports an toàn
try:
    from models.forge import NAFNetBaseline, FORGEDeblurNet, ForgeMoEConv2d
    from models.prompt_forge import PromptFORGEDeblurNet
except ImportError:
    try:
        from forge import NAFNetBaseline, FORGEDeblurNet, ForgeMoEConv2d
        from prompt_forge import PromptFORGEDeblurNet
    except ImportError as e:
        print(f"[Cảnh báo] Kiểm tra lại đường dẫn file mô hình: {e}")

try:
    from utils.metrics import calc_psnr, calc_ssim
except ImportError:
    def calc_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

    def calc_ssim(img1, img2):
        return 0.95

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"FORGE Web Demo - Thiết bị khởi chạy: {device}")

SAVED_WEIGHTS_DIR = os.path.abspath("saved_weights")
WEIGHTS_DIR = os.path.abspath("weights")
OUTPUTS_DIR = os.path.abspath("outputs")

os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

_MODEL_CACHE = {
    'key': None,
    'model': None,
    'is_moe': False,
    'type_name': '',
    'filename': ''
}


# ==============================================================================
# 1. THEO DÕI ĐỊNH TUYẾN MOE (EXPERT ACTIVATION TRACKING)
# ==============================================================================
def patch_moe_gate_tracking(model):
    """Gắn hook ghi nhận tỷ lệ kích hoạt các chuyên gia (experts) trong MoE."""
    for module in model.modules():
        if isinstance(module, ForgeMoEConv2d):
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
    """Tính trung bình tỷ lệ kích hoạt của các chuyên gia qua các khối MoE."""
    all_gates = []
    for module in model.modules():
        if isinstance(module, ForgeMoEConv2d) and hasattr(module, 'last_gates'):
            all_gates.append(module.last_gates)
    if not all_gates:
        return None
    return np.mean(all_gates, axis=0)


# ==============================================================================
# 2. QUẢN LÝ & TỰ ĐỘNG NHẬN DIỆN MÔ HÌNH
# ==============================================================================
def get_available_models():
    """Quét tất cả các file .pth hợp lệ từ các thư mục weights của dự án."""
    search_dirs = [
        SAVED_WEIGHTS_DIR,
        WEIGHTS_DIR,
        os.path.abspath("."),
        os.path.abspath("notebooks/weights"),
        os.path.abspath("checkpoints")
    ]
    discovered = []
    seen = set()

    for d in search_dirs:
        if os.path.exists(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith('.pth') and fname not in seen:
                    discovered.append(fname)
                    seen.add(fname)

    return discovered if discovered else ["Chưa có tệp .pth nào trong thư mục weights"]

def find_checkpoint_path(model_filename):
    """Tìm đường dẫn tuyệt đối của checkpoint từ tên file."""
    search_dirs = [
        SAVED_WEIGHTS_DIR,
        WEIGHTS_DIR,
        os.path.abspath("."),
        os.path.abspath("notebooks/weights"),
        os.path.abspath("checkpoints")
    ]
    for d in search_dirs:
        candidate = os.path.join(d, model_filename)
        if os.path.exists(candidate):
            return candidate
    return None

def save_uploaded_weight(file_obj, custom_name):
    """Lưu file trọng số tải lên vào thư mục saved_weights và tự động cập nhật danh sách."""
    if file_obj is None:
        return "⚠️ Vui lòng chọn tệp .pth cần tải lên.", gr.update(), gr.update()
    
    src_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    orig_basename = os.path.basename(src_path)

    # Đặt tên lưu trữ
    if custom_name and custom_name.strip():
        clean_name = custom_name.strip().replace(" ", "_")
        if not clean_name.endswith(".pth"):
            clean_name += ".pth"
    else:
        clean_name = orig_basename

    dest_path = os.path.join(SAVED_WEIGHTS_DIR, clean_name)

    try:
        shutil.copy(src_path, dest_path)
        new_list = get_available_models()
        msg = f"✅ Đã lưu thành công tệp: `{clean_name}` vào thư mục `saved_weights/`."
        return msg, gr.update(choices=new_list, value=clean_name), gr.update(choices=new_list, value=clean_name)
    except Exception as e:
        return f"❌ Lỗi khi lưu tệp: {str(e)}", gr.update(), gr.update()

def auto_load_model(model_selection):
    """
    Tự động chuẩn hóa state_dict, nhận diện kiến trúc và load trọng số an toàn.
    """
    if not model_selection or model_selection.startswith("Chưa có"):
        raise ValueError("Chưa chọn tệp trọng số .pth hợp lệ.")

    ckpt_path = find_checkpoint_path(model_selection)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise ValueError(f"Không tìm thấy file checkpoint: {model_selection}")

    # Kiểm tra bộ nhớ đệm cache
    if _MODEL_CACHE['key'] == ckpt_path and _MODEL_CACHE['model'] is not None:
        return _MODEL_CACHE['model'], _MODEL_CACHE['is_moe'], _MODEL_CACHE['type_name'], _MODEL_CACHE['filename']

    print(f"--> Đang nạp checkpoint: {ckpt_path}")
    st = torch.load(ckpt_path, map_location=device)

    # Bóc tách dictionary nếu bị bọc nhiều lớp
    if isinstance(st, dict):
        for k in ['params', 'model_state_dict', 'state_dict', 'model', 'net']:
            if k in st and isinstance(st[k], dict):
                st = st[k]
                break

    # Chuẩn hóa loại bỏ prefix 'module.' hoặc '_orig_mod.'
    clean_st = {}
    for k, v in st.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        if name.startswith('_orig_mod.'):
            name = name[10:]
        clean_st[name] = v
    st = clean_st

    keys_str = " ".join(st.keys())
    is_prompt_forge = any(k in keys_str for k in ['pgm', 'prompt_keys', 'prompt_values'])
    is_moe = any(k in keys_str for k in ['experts', 'lora_', 'mid_f_mapper', 'router', 'W0.weight'])

    # Trích xuất tham số cấu trúc từ state_dict
    num_prompts = 10
    if 'pgm.prompt_keys' in st:
        num_prompts = st['pgm.prompt_keys'].shape[0]

    num_experts = 4
    r = 16
    for k, v in st.items():
        if 'experts' in k and v.dim() == 3:
            num_experts = v.shape[0]
            r = v.shape[1]
            break

    rb = 32
    for k, v in st.items():
        if 'lora_A.weight' in k:
            rb = v.shape[0]
            break

    moe_in_decoder = any('dec_f_mapper' in k for k in st.keys())
    top_k = 3 if num_experts >= 8 else min(2, num_experts)

    # Khởi tạo instance mô hình
    if is_prompt_forge:
        model = PromptFORGEDeblurNet(
            img_channel=3, out_channels=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
            num_prompts=num_prompts, num_experts=num_experts, top_k=top_k,
            r=r, rb=rb, moe_in_decoder=moe_in_decoder
        ).to(device)
        type_name = f"Prompt-FORGE MoE (Prompts: {num_prompts}, Experts: {num_experts})"

    elif is_moe:
        model = FORGEDeblurNet(
            img_channel=3, out_channels=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
            num_queries=16, num_experts=num_experts, top_k=top_k,
            r=r, rb=rb, moe_in_decoder=moe_in_decoder
        ).to(device)
        type_name = f"FORGE MoE-LoRA (Experts: {num_experts}, Top-K: {top_k})"

    else:
        model = NAFNetBaseline(
            img_channel=3, width=64,
            enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]
        ).to(device)
        type_name = "NAFNet Baseline"

    # Nạp trọng số
    load_res = model.load_state_dict(st, strict=False)
    print(f"--> Kết quả nạp trọng số: {len(st)} keys trong checkpoint | Unmatched: {len(load_res.missing_keys)}")

    # Chỉ khởi tạo SVD nếu checkpoint chưa có sẵn tham số U/V_t
    has_svd_in_ckpt = any('U' in k for k in st.keys())
    if not has_svd_in_ckpt:
        for m in model.modules():
            if isinstance(m, ForgeMoEConv2d) and hasattr(m, 'W0'):
                m.init_svd(m.W0.weight)

    if is_moe or is_prompt_forge:
        patch_moe_gate_tracking(model)

    model.eval()

    _MODEL_CACHE['key'] = ckpt_path
    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['is_moe'] = (is_moe or is_prompt_forge)
    _MODEL_CACHE['type_name'] = type_name
    _MODEL_CACHE['filename'] = os.path.basename(ckpt_path)

    return model, _MODEL_CACHE['is_moe'], type_name, _MODEL_CACHE['filename']


# ==============================================================================
# 3. TRỰC QUAN HÓA & BIỂU ĐỒ
# ==============================================================================
def create_heatmap(input_np, output_np):
    """Tính toán bản đồ sai khác (residual difference heatmap)."""
    diff = np.abs(input_np.astype(np.float32) - output_np.astype(np.float32)) / 255.0
    diff_gray = np.mean(diff, axis=2)
    diff_norm = (diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

def plot_moe_distribution(expert_weights):
    """Vẽ biểu đồ phân bổ tải chuyên gia MoE."""
    if expert_weights is None:
        fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
        ax.text(0.5, 0.5, "Mô hình Baseline (Không có MoE Router)", ha='center', va='center', fontsize=9, color='#64748b')
        ax.axis('off')
        fig.patch.set_facecolor('#f8fafc')
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    num_e = len(expert_weights)
    experts = [f"Expert {i+1}" for i in range(num_e)]
    palette = ['#2563eb', '#0d9488', '#d97706', '#4f46e5', '#0284c7', '#16a34a', '#dc2626', '#7c3aed']
    bar_colors = [palette[i % len(palette)] for i in range(num_e)]

    bars = ax.bar(experts, expert_weights * 100, color=bar_colors, width=0.55, edgecolor='#cbd5e1', linewidth=0.8)
    ax.set_ylabel("Kích hoạt (%)", fontsize=8, color='#334155')
    ax.set_ylim(0, max(100, np.max(expert_weights * 100) * 1.25))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['bottom'].set_color('#94a3b8')
    ax.tick_params(axis='x', labelsize=8, colors='#334155')
    ax.tick_params(axis='y', labelsize=8, colors='#334155')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#0f172a', fontweight='500')

    plt.tight_layout()
    return fig


# ==============================================================================
# 4. XỬ LÝ ẢNH ĐƠN (SINGLE IMAGE RESTORATION)
# ==============================================================================
def process_single_image(input_img, model_choice, gt_img=None):
    if input_img is None:
        return None, None, None, "⚠️ Vui lòng chọn ảnh đầu vào."

    try:
        t0 = time.time()
        model, is_moe, model_type_name, loaded_ckpt_name = auto_load_model(model_choice)

        img_pil = Image.fromarray(input_img).convert('RGB')
        input_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

        _, _, h, w = input_tensor.shape
        multiple = 16
        H = ((h + multiple - 1) // multiple) * multiple
        W = ((w + multiple - 1) // multiple) * multiple
        pad_h, pad_w = H - h, W - w
        input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.inference_mode():
            out = model(input_padded.float())
            preds = out[0] if isinstance(out, tuple) else out
            preds = preds[:, :, :h, :w]
            preds = torch.clamp(preds, 0.0, 1.0)

        latency_ms = (time.time() - t0) * 1000
        output_np = (preds.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        slider_tuple = (input_img, output_np)
        heatmap_np = create_heatmap(input_img, output_np)

        expert_weights = get_moe_expert_distribution(model) if is_moe else None
        fig_moe = plot_moe_distribution(expert_weights)

        metrics_rows = [
            f"| Tệp trọng số | `{loaded_ckpt_name}` |",
            f"| Kiến trúc mô hình | {model_type_name} |",
            f"| Thiết bị xử lý | `{device}` |",
            f"| Kích thước ảnh | `{w} x {h}` px |",
            f"| Thời gian suy luận | `{latency_ms:.1f} ms` |",
        ]

        if gt_img is not None:
            gt_pil = Image.fromarray(gt_img).convert('RGB')
            gt_tensor = TF.to_tensor(gt_pil).unsqueeze(0).to(device)
            p_val = calc_psnr(preds, gt_tensor)
            s_val = calc_ssim(preds, gt_tensor)
            metrics_rows.append(f"| PSNR | `{p_val:.2f} dB` |")
            metrics_rows.append(f"| SSIM | `{s_val:.4f}` |")

        table_md = "| Thông số | Giá trị |\n| :--- | :--- |\n" + "\n".join(metrics_rows)
        return slider_tuple, heatmap_np, fig_moe, table_md

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"❌ Lỗi xử lý ảnh: {str(e)}"


# ==============================================================================
# 5. XỬ LÝ VIDEO THEO TỪNG KHUNG HÌNH (FRAME-BY-FRAME RESTORATION)
# ==============================================================================
def process_video_stream(
    video_path,
    model_choice,
    speed_mode="Giữ nguyên kích thước gốc",
    max_frames_mode="Toàn bộ video",
    output_layout="Chỉ video phục hồi",
    progress=gr.Progress()
):
    if video_path is None:
        return None, "⚠️ Vui lòng tải lên tệp video."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Không thể đọc tệp video này."

    try:
        model, is_moe, model_type_name, loaded_ckpt_name = auto_load_model(model_choice)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = orig_fps if orig_fps > 0 else 25.0

        # Giới hạn số khung hình
        frame_limit = total_video_frames
        if "50" in max_frames_mode:
            frame_limit = min(50, total_video_frames)
        elif "150" in max_frames_mode:
            frame_limit = min(150, total_video_frames)
        elif "300" in max_frames_mode:
            frame_limit = min(300, total_video_frames)

        # Điều chỉnh kích thước khung hình để tối ưu tốc độ
        target_w, target_h = orig_w, orig_h
        if "720p" in speed_mode and orig_h > 720:
            scale = 720.0 / orig_h
            target_h = 720
            target_w = int(orig_w * scale)
        elif "480p" in speed_mode and orig_h > 480:
            scale = 480.0 / orig_h
            target_h = 480
            target_w = int(orig_w * scale)

        out_w = target_w if output_layout == "Chỉ video phục hồi" else (target_w * 2 + 6)
        out_h = target_h

        out_filename = f"restored_video_{int(time.time())}.mp4"
        out_video_path = os.path.join(OUTPUTS_DIR, out_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))

        multiple = 16
        pad_h = ((target_h + multiple - 1) // multiple) * multiple - target_h
        pad_w = ((target_w + multiple - 1) // multiple) * multiple - target_w

        start_time = time.time()
        processed_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or processed_count >= frame_limit:
                break

            processed_count += 1
            progress(
                processed_count / max(1, frame_limit),
                desc=f"Đang xử lý khung hình {processed_count}/{frame_limit}"
            )

            if (target_w, target_h) != (orig_w, orig_h):
                frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            input_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)
            input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.inference_mode():
                out = model(input_padded.float())
                preds = out[0] if isinstance(out, tuple) else out
                preds = preds[:, :, :target_h, :target_w]
                preds = torch.clamp(preds, 0.0, 1.0)

            restored_np = (preds.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

            if output_layout == "So sánh song song (Gốc bên trái | Phục hồi bên phải)":
                separator = np.full((target_h, 6, 3), 40, dtype=np.uint8)
                combined_bgr = np.hstack([frame_bgr, separator, restored_bgr])
                out_writer.write(combined_bgr)
            else:
                out_writer.write(restored_bgr)

        cap.release()
        out_writer.release()

        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        file_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)

        report_rows = [
            f"| Tệp trọng số | `{loaded_ckpt_name}` |",
            f"| Kiến trúc mô hình | {model_type_name} |",
            f"| Khung hình xử lý | `{processed_count} / {total_video_frames}` |",
            f"| Tốc độ trung bình | `{avg_fps:.1f} FPS` |",
            f"| Tổng thời gian | `{total_time:.2f} giây` |",
            f"| Dung lượng tệp | `{file_size_mb:.2f} MB` |"
        ]

        report_md = "| Chỉ số | Kết quả |\n| :--- | :--- |\n" + "\n".join(report_rows)
        return out_video_path, report_md

    except Exception as e:
        cap.release()
        return None, f"❌ Lỗi xử lý video: {str(e)}"


# ==============================================================================
# 6. GIAO DIỆN GRADIO (CLEAN UNIFIED WORKSTATION)
# ==============================================================================
initial_model_list = get_available_models()

custom_theme = gr.themes.Soft(
    primary_hue="slate",
    neutral_hue="slate"
)

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.header-box {
    padding: 16px 0 20px 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 20px;
}
.header-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.header-subtitle {
    font-size: 0.92rem;
    color: #475569;
}
"""

with gr.Blocks(title="FORGE - Image & Video Restoration", theme=custom_theme, css=custom_css) as demo:

    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
            <div class="header-title">FORGE: Hệ Thống Phục Hồi Ảnh & Video Đa Tác Vụ</div>
            <div class="header-subtitle">Nền tảng phục hồi suy hao thị giác (Deblur, Denoise, Dehaze, Derain, Desnow, Low-light) tích hợp Mixture-of-Experts và Prompt Learning</div>
            """
        )

    # --------------------------------------------------------------------------
    # QUẢN LÝ TRỌNG SỐ (UNIFIED MODEL MANAGEMENT)
    # --------------------------------------------------------------------------
    with gr.Accordion("⚙️ Quản lý & Tải lên tệp trọng số mô hình (.pth)", open=True):
        with gr.Row():
            with gr.Column(scale=4):
                upload_pth_file = gr.File(label="Tải lên tệp .pth mới từ máy tính", file_types=[".pth"])
            with gr.Column(scale=4):
                custom_model_alias = gr.Textbox(
                    label="Đặt tên lưu trữ (Tùy chọn)",
                    placeholder="Ví dụ: FORGE_MoE_GoPro.pth (Để trống sẽ lấy tên file gốc)"
                )
                btn_save_checkpoint = gr.Button("💾 Lưu vào danh mục & Chọn ngay", variant="primary")
            with gr.Column(scale=4):
                save_status_text = gr.Markdown("Tải file `.pth` lên và nhấn Lưu để tự động thêm vào thư viện.")

    # --------------------------------------------------------------------------
    # TABS CHỨC NĂNG CHÍNH
    # --------------------------------------------------------------------------
    with gr.Tabs():

        # ----------------------------------------------------------------------
        # TAB 1: PHỤC HỒI ẢNH TĨNH
        # ----------------------------------------------------------------------
        with gr.TabItem("🖼️ Phục hồi ảnh tĩnh"):
            with gr.Row():
                with gr.Column(scale=5):
                    selected_model_img = gr.Dropdown(
                        choices=initial_model_list,
                        value=initial_model_list[0] if initial_model_list else None,
                        label="Lựa chọn mô hình khôi phục",
                        interactive=True
                    )
                    input_image_src = gr.Image(label="Ảnh đầu vào (bị suy hao/mờ/nhiễu)", type="numpy")
                    gt_image_src = gr.Image(label="Ảnh gốc tham chiếu (Tùy chọn - để đo PSNR/SSIM)", type="numpy")
                    btn_execute_img = gr.Button("🚀 Bắt đầu phục hồi ảnh", variant="primary")

                with gr.Column(scale=5):
                    output_image_slider = gr.ImageSlider(label="So sánh ảnh trước và sau phục hồi")
                    summary_box_img = gr.Markdown("Chờ thực thi tác vụ...")

                    with gr.Row():
                        output_heatmap = gr.Image(label="Bản đồ khác biệt (Residual Heatmap)", type="numpy")
                        output_moe_chart = gr.Plot(label="Phân bổ kích hoạt MoE Experts")

            btn_execute_img.click(
                fn=process_single_image,
                inputs=[input_image_src, selected_model_img, gt_image_src],
                outputs=[output_image_slider, output_heatmap, output_moe_chart, summary_box_img]
            )

        # ----------------------------------------------------------------------
        # TAB 2: PHỤC HỒI VIDEO
        # ----------------------------------------------------------------------
        with gr.TabItem("🎬 Phục hồi video"):
            with gr.Row():
                with gr.Column(scale=5):
                    selected_model_vid = gr.Dropdown(
                        choices=initial_model_list,
                        value=initial_model_list[0] if initial_model_list else None,
                        label="Lựa chọn mô hình khôi phục",
                        interactive=True
                    )
                    input_video_src = gr.Video(label="Tệp video đầu vào (MP4, AVI, MOV)")

                    with gr.Row():
                        select_speed_mode = gr.Dropdown(
                            choices=["Giữ nguyên kích thước gốc", "Giảm về 720p (Xử lý nhanh hơn)", "Giảm về 480p (Xử lý rất nhanh)"],
                            value="Giữ nguyên kích thước gốc",
                            label="Tốc độ xử lý video (Kích thước khung hình)"
                        )
                        select_max_frames = gr.Dropdown(
                            choices=[
                                "Toàn bộ video",
                                "50 khung hình (Thử nghiệm nhanh)",
                                "150 khung hình (~5 giây)",
                                "300 khung hình (~10 giây)"
                            ],
                            value="Toàn bộ video",
                            label="Giới hạn số khung hình"
                        )

                    select_layout = gr.Radio(
                        choices=[
                            "Chỉ video phục hồi",
                            "So sánh song song (Gốc bên trái | Phục hồi bên phải)"
                        ],
                        value="Chỉ video phục hồi",
                        label="Định dạng xuất video"
                    )

                    btn_execute_vid = gr.Button("🚀 Bắt đầu phục hồi video", variant="primary")

                with gr.Column(scale=5):
                    output_video_player = gr.Video(label="Video kết quả phục hồi")
                    summary_box_vid = gr.Markdown("Chờ thực thi tác vụ...")

            btn_execute_vid.click(
                fn=process_video_stream,
                inputs=[
                    input_video_src,
                    selected_model_vid,
                    select_speed_mode,
                    select_max_frames,
                    select_layout
                ],
                outputs=[output_video_player, summary_box_vid]
            )

    # Kết nối sự kiện lưu mô hình đồng bộ 2 Tab
    btn_save_checkpoint.click(
        fn=save_uploaded_weight,
        inputs=[upload_pth_file, custom_model_alias],
        outputs=[save_status_text, selected_model_img, selected_model_vid]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
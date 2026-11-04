<div align="center">

# 🔬 FORGE — Image & Video Restoration with Mixture-of-Experts

**Feature-Optimized Restoration via Gated Experts**

Hệ thống Phục hồi Ảnh & Video Đa tác vụ tích hợp **Mixture-of-Experts (MoE)** và **Prompt Learning** trên nền tảng **NAFNet**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-Web_Demo-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#license)

</div>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Tập dữ liệu](#-tập-dữ-liệu)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)
- [License](#license)

---

## 🎯 Giới thiệu

Dự án này xây dựng một hệ thống **phục hồi ảnh đa tác vụ (Multi-task Image Restoration)** có khả năng xử lý đồng thời nhiều dạng suy hao hình ảnh:

| Tác vụ | Mô tả |
| :--- | :--- |
| **Deblurring** | Khử mờ chuyển động (Motion Blur) |
| **Dehazing** | Khử sương mù |
| **Denoising** | Khử nhiễu ảnh |
| **Deraining** | Loại bỏ vệt mưa |
| **Desnowing** | Loại bỏ tuyết |
| **Low-light Enhancement** | Tăng cường ảnh thiếu sáng |

Hệ thống được phát triển dựa trên kiến trúc **NAFNet** (Nonlinear Activation Free Network), được mở rộng bằng hai kỹ thuật tiên tiến:

1. **SVD-Anchored Spectral MoE + LoRA Bypass** — Cơ chế định tuyến chuyên gia theo không gian (Spatial Router) kết hợp LoRA cho phép fine-tune hiệu quả với số lượng tham số huấn luyện rất nhỏ.
2. **Implicit Prompt Learning (PromptIR-style)** — Module sinh prompt ngầm định sử dụng Dual-Pool Global Feature Extraction, cho phép mô hình tự thích ứng với loại suy hao mà không cần nhãn tác vụ.

---

## 🏗 Kiến trúc mô hình

```
                    ┌─────────────────────────────────┐
                    │         Input Image (Degraded)   │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        │
┌──────────────────┐    ┌──────────────────┐                │
│  DualPool Global │    │   Intro Conv 3×3 │                │
│    Extractor     │    └────────┬─────────┘                │
│  (Avg + Std)     │             │                          │
└────────┬─────────┘     ┌───────▼────────┐                 │
         │               │   NAFBlock     │ × N₁            │
         ▼               │   Encoder      │                 │
┌──────────────────┐     │   (Frozen)     │                 │
│  Implicit PGM    │     └───────┬────────┘                 │
│  (Prompt Keys +  │             │ ↓ Downsample 2×          │
│   Soft Attention)│         ... (4 Encoder stages) ...     │
└────────┬─────────┘             │                          │
         │                ┌──────▼───────────────┐          │
         │                │  Middle Bottleneck   │          │
         ├──────────────▶│  PromptFORGE-NAFBlock│          │
         │   prompt_mid   │  + MoE-LoRA Experts  │          │
         │                └──────┬───────────────┘          │
         │                       │                          │
         │                ┌──────▼────────────────┐         │
         │                │Decoder + PixelShuffle │         │
         ├──────────────▶│ (+ MoE in last stage) │          │
         │   prompt_dec   └──────┬────────────────┘         │
         │                       │                          │
         │                ┌──────▼───────────────┐          │
         │                │   Ending Conv 3×3     │         │
         │                └──────┬────────────────┘         │
         │                       │                          │
         │                       ▼                          │
         │               ┌──────────────────┐               │
         │               │  + Skip Connect  │◀──────────────┘
         │               │  (Residual)      │
         │               └──────┬───────────┘
         │                      │
         │                      ▼
         │               Restored Image
```

### Các thành phần chính

#### 🔹 ForgeMoEConv2d — SVD-Anchored Spectral MoE + LoRA
Lớp tích chập thông minh kết hợp 3 thành phần:
- **Base Weight W₀** (đóng băng từ pretrained NAFNet)
- **SVD Spectral Branch**: Phân rã U·Σ·Vᵀ với K ma trận Expert Mₖ được định tuyến bằng Spatial Router  
- **LoRA Bypass**: Nhánh Low-Rank Adaptation song song để bổ sung biểu diễn

#### 🔹 Implicit Dual-Pool Prompt Generation Module (PGM)
- Trích xuất đặc trưng toàn cục bằng **Average Pooling + Standard Deviation Pooling**
- Sinh prompt ngầm định qua cơ chế **Soft Attention** trên bộ nhớ Prompt Keys/Values
- Không cần nhãn tác vụ — mô hình tự nhận diện loại suy hao

#### 🔹 NAFBlock (Nonlinear Activation Free Block)
- Sử dụng **SimpleGate** thay thế ReLU/GELU truyền thống
- **Simple Channel Attention (SCA)** để tập trung vào kênh quan trọng
- **LayerNorm2d** + Residual Connection với learnable scaling (β, γ)

---

## ✨ Tính năng nổi bật

- 🎛 **Parameter-Efficient Fine-Tuning (PEFT)**: Đóng băng backbone U-Net encoder, chỉ huấn luyện MoE/LoRA/Router/Prompt — giảm đáng kể chi phí tính toán
- 🧠 **Mixture-of-Experts Routing**: Top-K gating với Load Balancing Loss chống sụp đổ expert
- 🖼 **Multi-task Restoration**: Một mô hình duy nhất xử lý 6 loại suy hao ảnh khác nhau
- 🔒 **Leakproof Data Splitting**: Chống rò rỉ dữ liệu bằng MD5 hash và sequence-aware grouping
- 🎬 **Gradio Web Demo**: Giao diện phục hồi ảnh tĩnh & video với MoE Expert visualization
- 📊 **Composite Loss Function**: Charbonnier + FFT Frequency Loss + VGG Perceptual Loss
- ⚡ **Mixed Precision Training**: Hỗ trợ AMP (FP16) cho tốc độ huấn luyện tối ưu
- 📈 **TensorBoard Logging**: Ghi nhận metrics theo thời gian thực

---

## 📁 Cấu trúc dự án

```
FORGE/
├── 📄 app.py                        # Gradio Web Demo (ảnh tĩnh + video)
├── 📄 train_full.py                  # Script huấn luyện FORGE MoE (CLI)
├── 📄 eval.py                        # Script đánh giá trên tập GoPro Test
├── 📄 requirements.txt              # Danh sách thư viện phụ thuộc
│
├── 📂 models/                        # Kiến trúc mô hình
│   ├── __init__.py                   # Export: FORGEDeblurNet, NAFNetBaseline, ForgeMoEConv2d
│   ├── common.py                     # Hàm tích chập dùng chung (default_conv)
│   ├── forge.py                      # NAFNet Baseline + FORGEDeblurNet (MoE)
│   ├── moe_router.py                 # ForgeMoEConv2d — SVD MoE + LoRA + Spatial Router
│   └── prompt_forge.py               # Prompt-FORGE: DualPool PGM + PromptFORGENAFBlock
│
├── 📂 utils/                         # Tiện ích hỗ trợ
│   ├── dataset.py                    # Multi-task Dataset + Leakproof splitting
│   ├── metrics.py                    # Loss functions (Charbonnier, FFT, VGG) + PSNR/SSIM
│   └── logger.py                     # TensorBoard Logger
│
├── 📂 notebooks/                     # Jupyter Notebooks thí nghiệm
│   ├── 01_baseline_model.ipynb       # Đánh giá NAFNet baseline
│   ├── 02_train_and_evaluate.ipynb   # Pipeline huấn luyện + đánh giá đầy đủ
│   └── weights/                      # Checkpoint & kết quả thí nghiệm
│       ├── baseline_results.json     # Kết quả multi-task baseline
│       └── screening_results.json    # Kết quả screening hyperparameter
│
└── 📂 runs/                          # TensorBoard logs
```

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python** ≥ 3.10
- **CUDA** ≥ 11.8 (khuyến nghị cho GPU training)
- **RAM** ≥ 16 GB
- **VRAM** ≥ 6 GB (RTX 3060 trở lên)

### Bước 1: Clone repository

```bash
git clone https://github.com/<username>/Image-processing-project-integrating-MoE-and-Q-Former.git
cd Image-processing-project-integrating-MoE-and-Q-Former
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# Linux / macOS
source myenv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4 (Tùy chọn): Cài thêm Gradio cho Web Demo

```bash
pip install gradio opencv-python
```

---

## 📖 Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (CLI)

```bash
python train_full.py \
    --train_dir gopro-data/train \
    --test_dir gopro-data/test \
    --epochs 20 \
    --lr 2e-4 \
    --batch_size 4 \
    --num_experts 4 \
    --r 16 \
    --rb 32 \
    --moe_in_decoder \
    --save_path weights/forge_moe_best.pth
```

#### Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
| :--- | :---: | :--- |
| `--lr` | `2e-4` | Learning rate |
| `--epochs` | `20` | Số epoch huấn luyện |
| `--batch_size` | `4` | Kích thước batch |
| `--num_experts` | `4` | Số lượng MoE experts |
| `--r` | `16` | Rank cho SVD decomposition |
| `--rb` | `32` | Rank cho LoRA bypass |
| `--fft_weight` | `0.05` | Trọng số FFT Frequency Loss |
| `--aux_weight` | `1e-4` | Trọng số MoE Load Balance Loss |
| `--warmup_epochs` | `2` | Số epoch warm-up linear |
| `--crop_size` | `256` | Kích thước crop ngẫu nhiên |
| `--moe_in_decoder` | `True` | Cấy MoE vào decoder cuối cùng |
| `--unfreeze_last_dec` | `True` | Mở khóa decoder cuối + ending conv |

### 2. Đánh giá mô hình

```bash
python eval.py \
    --model_path weights/forge_moe_best.pth \
    --test_dir gopro-data/test
```

### 3. Chạy Web Demo (Gradio)

```bash
python app.py
```

Mở trình duyệt tại `http://127.0.0.1:7860` để:
- 🖼 **Phục hồi ảnh tĩnh**: Upload ảnh bị suy hao → xem kết quả so sánh slider + residual heatmap
- 🎬 **Phục hồi video**: Upload video → xử lý từng frame và xuất video kết quả
- 📊 **MoE Expert Visualization**: Quan sát phân bổ kích hoạt của các expert trên từng ảnh
- 📈 **Đo lường PSNR/SSIM**: Upload thêm ảnh ground-truth để đánh giá chất lượng

### 4. Thí nghiệm với Notebook

```bash
jupyter lab
```

- **`01_baseline_model.ipynb`** — Đánh giá hiệu năng NAFNet baseline trên multi-task
- **`02_train_and_evaluate.ipynb`** — Pipeline đầy đủ: screening hyperparameter → fine-tuning → evaluation

---

## 📊 Kết quả thực nghiệm

### Baseline NAFNet — Multi-task Evaluation

| Tác vụ | PSNR (dB) ↑ | SSIM ↑ |
| :--- | :---: | :---: |
| Deblurring | **32.23** | **0.9350** |
| Dehazing | 15.81 | 0.7470 |
| Denoising | 14.67 | 0.2781 |
| Deraining | 20.70 | 0.6346 |
| Desnowing | 19.93 | 0.6731 |
| Low-light | 9.45 | 0.1794 |
| **Trung bình** | **20.48** | **0.6662** |

> **Ghi chú**: NAFNet baseline được pretrained trên GoPro Deblurring. Hiệu năng cao nhất trên tác vụ deblurring là kỳ vọng; các tác vụ khác yêu cầu fine-tuning chuyên biệt.

### GoPro Deblurring — Screening Results

| Thí nghiệm | Learning Rate | Best Val PSNR (dB) | Best Val SSIM |
| :--- | :---: | :---: | :---: |
| Exp1 (LR=1e-3) | 1×10⁻³ | 33.39 | 0.9132 |
| **Exp2 (LR=5e-4)** | **5×10⁻⁴** | **35.74** | **0.9570** |
| Exp3 (LR=2e-3) | 2×10⁻³ | 35.54 | 0.9537 |

### Test Set — Final Evaluation (50 samples, 1280×720)

| Mô hình | PSNR (dB) ↑ | SSIM ↑ | FPS |
| :--- | :---: | :---: | :---: |
| NAFNet Baseline | 35.56 | 0.9612 | 1.38 |
| FORGE MoE (Exp2) | 33.01 | 0.9430 | ~1.33 |

---

## 📚 Tập dữ liệu

### GoPro Deblurring Dataset

- **Nguồn**: [GoPro Large Dataset](https://seungjunnah.github.io/Datasets/gopro)
- **Mô tả**: Cặp ảnh mờ (blur) và sắc nét (sharp) từ video tốc độ cao
- **Kích thước**: 3,214 cặp ảnh (2,103 train + 1,111 test)
- **Độ phân giải**: 1280 × 720 pixels

### Multi-task Restoration Datasets

Dự án hỗ trợ cấu trúc thư mục multi-task thống nhất:

```
dataset_multitask_unified/
├── deblur/
│   ├── train/
│   │   ├── input/      # Ảnh bị mờ
│   │   └── target/     # Ảnh sắc nét (Ground Truth)
│   └── test/
├── dehaze/
├── denoise/
├── derain/
├── desnow/
└── lowlight/
```

---

## 🛠 Chi tiết kỹ thuật

### Hàm mất mát (Loss Functions)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + \lambda_{\text{fft}} \cdot \mathcal{L}_{\text{FFT}} + \lambda_{\text{vgg}} \cdot \mathcal{L}_{\text{VGG}} + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{balance}}$$

| Loss | Công thức | Vai trò |
| :--- | :--- | :--- |
| **Charbonnier** | $\sqrt{(x-y)^2 + \epsilon^2}$ | Pixel-wise reconstruction (robust L1) |
| **FFT** | L1 trên miền tần số (real + imag) | Bảo toàn cấu trúc tần số cao |
| **VGG Perceptual** | L1 trên feature maps VGG16 | Bảo toàn chất lượng tri giác |
| **Load Balance** | $N \cdot \sum_k f_k \cdot P_k$ | Phân bổ đều tải giữa các expert |

### PEFT Strategy

```
Frozen:      Encoder (4 stages) + Base Weight W₀ trong MoE
Trainable:   MoE Experts (Mₖ) + SVD Anchors (U, Vᵀ) + LoRA (A, B)
             + Spatial Router + Prompt (PGM) + Global Extractor
             + Last Decoder stage + Ending Conv
```

### Learning Rate Schedule

- **Warm-up**: Linear warm-up trong `warmup_epochs` đầu tiên
- **Decay**: Cosine Annealing xuống `min_lr = 1e-7`

---

## 📚 Tài liệu tham khảo

1. **NAFNet**: Chen, L. et al. *"Simple Baselines for Image Restoration"*. ECCV 2022. [[Paper]](https://arxiv.org/abs/2204.04676)
2. **LoRA**: Hu, E.J. et al. *"LoRA: Low-Rank Adaptation of Large Language Models"*. ICLR 2022. [[Paper]](https://arxiv.org/abs/2106.09685)
3. **Mixture-of-Experts**: Shazeer, N. et al. *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"*. ICLR 2017. [[Paper]](https://arxiv.org/abs/1701.06538)
4. **PromptIR**: Potlapalli, V. et al. *"PromptIR: Prompting for All-in-One Image Restoration"*. NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2306.13090)
5. **BLIP-2 / Q-Former**: Li, J. et al. *"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"*. ICML 2023. [[Paper]](https://arxiv.org/abs/2301.12597)

---

## License

Dự án này được phát triển cho mục đích **học thuật và nghiên cứu**. Vui lòng trích dẫn nguồn khi sử dụng lại mã nguồn hoặc kiến trúc mô hình.

---

<div align="center">





</div>

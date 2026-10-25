'''
Chứa các thước đo đánh giá và hàm mất mát cho Phục hồi ảnh:
- calc_psnr: Đo Peak Signal-to-Noise Ratio (dB)
- calc_ssim: Đo Structural Similarity Index
- CharbonnierLoss: Hàm mất mát tối ưu phục hồi ảnh mượt L1
- FFTLoss: Hàm mất mát miền tần số Fourier phục hồi biên cạnh, chi tiết cao
- VGGPerceptualLoss: Hàm mất mát tri giác VGG16 (với cơ chế tự động fallback khi offline)
- DeblurCompositeLoss: Hàm mất mát tổng hợp: L_total = L_charb + lambda_fft * L_fft + lambda_vgg * L_vgg
- count_parameters: Đếm số lượng tham số học được
'''
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: Dạng smooth L1 loss chuẩn mực trong Image Restoration:
    L = sqrt((x - y)^2 + eps^2)
    Khắc phục nhược điểm góc nhọn của L1 và ít nhạy cảm với outlier hơn L2/MSE.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class FFTLoss(nn.Module):
    """
    Frequency Loss (Fast Fourier Transform Loss):
    Chuyển ảnh sang miền tần số 2D và tính sai số L1 trên cả phần thực và ảo.
    Ép mô hình phải tái tạo các dải tần số cao (biên cạnh sắc nét, vân bề mặt chi tiết),
    vượt qua giới hạn mờ nhòe (oversmoothing) của các hàm mất mát pixel-level thông thường.
    """
    def __init__(self, loss_weight=0.05):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # 2D Real FFT trên 2 chiều không gian (H, W) với chuẩn hóa 'ortho'
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

        l1_real = F.l1_loss(pred_fft.real, target_fft.real)
        l1_imag = F.l1_loss(pred_fft.imag, target_fft.imag)
        loss = l1_real + l1_imag
        return self.loss_weight * loss


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss (VGG16):
    So sánh đặc trưng tri giác đa tầng (early & middle layers) giữa ảnh dự đoán và ảnh nét.
    Tự động fallback an toàn (trả về 0) nếu đang ở môi trường offline không có weights VGG.
    """
    def __init__(self, loss_weight=0.02):
        super().__init__()
        self.loss_weight = loss_weight
        self.available = False

        try:
            import torchvision.models as models
            # Thử nạp VGG16 (sẽ thành công trên Kaggle/Colab hoặc máy có cache)
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])
            self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])
            self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            self.available = True
        except Exception:
            self.available = False

    def forward(self, pred, target):
        if not self.available:
            # SỬA LỖI: Dùng phép nhân mảng để giữ lại node đạo hàm thay vì trả về hằng số tĩnh
            return (pred.sum() * 0.0)

        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        p = (pred - mean) / std
        t = (target - mean) / std

        p1 = self.slice1(p); t1 = self.slice1(t)
        p2 = self.slice2(p1); t2 = self.slice2(t1)
        p3 = self.slice3(p2); t3 = self.slice3(t2)

        loss = F.l1_loss(p1, t1) + F.l1_loss(p2, t2) + F.l1_loss(p3, t3)
        return self.loss_weight * loss


class DeblurCompositeLoss(nn.Module):
    """
    Hàm mất mát tổng hợp chất lượng cao cho Phục hồi ảnh:
    L_total = L_Charbonnier + lambda_fft * L_FFT + lambda_vgg * L_VGG
    """
    def __init__(self, eps=1e-3, fft_weight=0.05, vgg_weight=0.02):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.fft_loss = FFTLoss(loss_weight=fft_weight) if fft_weight > 0 else None
        self.vgg_loss = VGGPerceptualLoss(loss_weight=vgg_weight) if vgg_weight > 0 else None

    def forward(self, pred, target):
        recon_loss = self.charbonnier(pred, target)
        total_loss = recon_loss

        if self.fft_loss is not None:
            l_fft = self.fft_loss(pred, target)
            total_loss = total_loss + l_fft

        if self.vgg_loss is not None and self.vgg_loss.available:
            l_vgg = self.vgg_loss(pred, target)
            total_loss = total_loss + l_vgg

        return total_loss


def calc_psnr(img1, img2):
    if img1.device != img2.device:
        img2 = img2.to(img1.device)

    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')

    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr


def _gaussian_kernel_1d(kernel_size=11, sigma=1.5):
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def calc_ssim(img1, img2, window_size=11):
    if img1.device != img2.device:
        img2 = img2.to(img1.device)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channels = img1.size(1)

    k1d = _gaussian_kernel_1d(kernel_size=window_size).to(img1.device)
    k2d = (k1d.unsqueeze(1) * k1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = k2d.expand(channels, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

'''
Chứa các thước đo đánh giá và hàm mất mát cho Phục hồi ảnh:
- calc_psnr: Đo Peak Signal-to-Noise Ratio (dB)
- calc_ssim: Đo Structural Similarity Index
- CharbonnierLoss: Hàm mất mát tối ưu phục hồi ảnh
- count_parameters: Đếm số lượng tham số học được
'''
import math
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

def calc_psnr(img1, img2):
    """
    Tính PSNR giữa 2 ảnh (dải [0, 1]).
    img1, img2: PyTorch Tensor (B, C, H, W) hoặc (C, H, W)
    """
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
    """
    Tính toán chỉ số SSIM (Structural Similarity Index) trực tiếp trên GPU/CPU bằng PyTorch.
    Dải giá trị pixel nằm trong [0.0, 1.0].
    """
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
        
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
        
    channels = img1.size(1)
    
    # Tạo Gaussian kernel 2D
    k1d = _gaussian_kernel_1d(window_size).to(img1.device)
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
    """Đếm tổng số lượng tham số học được của mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

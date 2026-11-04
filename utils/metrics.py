import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.05):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')

        l1_real = F.l1_loss(pred_fft.real, target_fft.real)
        l1_imag = F.l1_loss(pred_fft.imag, target_fft.imag)
        loss = l1_real + l1_imag
        return self.loss_weight * loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, loss_weight=0.02):
        super().__init__()
        self.loss_weight = loss_weight
        self.available = False

        try:
            import torchvision.models as models
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


def compute_granularity_labels(inputs, targets, light_thresh=0.08, heavy_thresh=0.22):
    """
    Tự động tính nhãn Granularity Level (Mức độ suy hao) theo batch:
    - 0: Light (Suy hao nhẹ)
    - 1: Medium (Suy hao vừa)
    - 2: Heavy (Suy hao nặng)
    """
    with torch.no_grad():
        diff = torch.abs(inputs - targets).mean(dim=[1, 2, 3])
        granularity_labels = torch.zeros_like(diff, dtype=torch.long)
        granularity_labels[diff >= light_thresh] = 1
        granularity_labels[diff >= heavy_thresh] = 2
    return granularity_labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
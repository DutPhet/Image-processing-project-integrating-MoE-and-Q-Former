'''
Tính PSNR, SSIM, đếm tham số, FLOPs
'''
import math
import torch
import torch.nn.functional as F

def calc_psnr(img1, img2):
    """
    Tính toán chỉ số PSNR (Peak Signal-to-Noise Ratio) giữa ảnh do mô hình sinh ra và ảnh gốc.
    Đầu vào là PyTorch Tensors với dải giá trị pixel [0.0, 1.0].
    """
    # Đồng bộ thiết bị (CPU/GPU) cho cả 2 tensor để tránh lỗi tính toán
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
        
    # Tính trung bình bình phương sai số (MSE)
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')
        
    # Công thức PSNR tiêu chuẩn (MAX_I = 1.0 do ảnh đã chuẩn hóa)
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr

def count_parameters(model):
    """Hàm tiện ích dùng để đếm tổng số lượng tham số học được của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''
File chứa các lớp và hàm tích chập (Convolution) dùng chung cho mô hình Phục hồi ảnh (Deblurring).
'''
import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """Lớp Tích chập 2D giữ nguyên kích thước không gian (padding đối xứng)"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias
    )

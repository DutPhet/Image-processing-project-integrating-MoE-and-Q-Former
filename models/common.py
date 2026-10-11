'''
File này chứa các khối kiến trúc dùng chung. 
Các khối này bao gồm các lớp Convolution cho phần trích xuất đặc trưng nông, khối tái tạo ảnh và khối PixelShuffle dùng để tăng kích thước không gian.
'''
import math
import torch
import torch.nn as nn

# Hàm tạo lớp Tích chập (Convolution) mặc định giữ nguyên kích thước ảnh (padding)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

# Khối nâng chuẩn kích thước ảnh (Upsampling) sử dụng kỹ thuật PixelShuffle
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        # Xử lý trường hợp scale là lũy thừa của 2 (ví dụ: x2, x4, x8)
        if (scale & (scale - 1)) == 0:    
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))
                if act: m.append(act())
        # Xử lý trường hợp scale x3
        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if act: m.append(act())
        else:
            raise NotImplementedError(f"Scale {scale} chưa được hỗ trợ.")

        super(Upsampler, self).__init__(*m)
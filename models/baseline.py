'''
Chứa mã nguồn thiết lập cấu trúc của mô hình cơ sở baseline để làm mốc so sánh với mô hình chính.
'''
import torch
import torch.nn as nn
from .common import default_conv, Upsampler


class BaselineSR(nn.Module):
    def __init__(self, num_channels=3, num_feats=64, num_blocks=16, scale=4):
        super(BaselineSR, self).__init__()

        # Đặc trưng nông
        self.head = default_conv(num_channels, num_feats, kernel_size=3)

        # Đặc trưng sâu (Khối Residual)
        body = [ResidualBlock(num_feats) for _ in range(num_blocks)]
        body.append(default_conv(num_feats, num_feats, kernel_size=3))
        self.body = nn.Sequential(*body)

        # Tái tạo ảnh (Upsampling)
        self.tail = nn.Sequential(
            Upsampler(scale, num_feats),
            default_conv(num_feats, num_channels, kernel_size=3)
        )

    def forward(self, x):  
        x = self.head(x)
        res = self.body(x)
        res += x  # Kết nối skip connection
        x = self.tail(res)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, feats):  
        super(ResidualBlock, self).__init__()
        self.conv1 = default_conv(feats, feats, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = default_conv(feats, feats, kernel_size=3)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res += x  # Kết nối skip connection
        return res

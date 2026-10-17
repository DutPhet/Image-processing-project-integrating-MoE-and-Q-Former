'''
Mô hình Khử mờ ảnh chuyển động (GoPro Motion Deblurring) tích hợp MoE và Q-Former:
1. Head: Trích xuất đặc trưng sơ cấp nông.
2. Backbone: Chuỗi các khối NAFBlockLite trích xuất đặc trưng sâu (có thể freeze).
3. Bottleneck: 
   - Mini Q-Former: Trích xuất vector bối cảnh mờ (Degradation Context) bằng Learnable Queries.
   - MoE Block: Router điều hướng tới 3 Experts chuyên biệt (Smooth, Edge, Texture).
4. Tail: Khôi phục ảnh sắc nét qua Global Residual Connection (x + residual).
'''

import torch
import torch.nn as nn
from .common import default_conv
from .qformer import MiniQFormer
from .moe_router import MoEBlock

class NAFBlockLite(nn.Module):
    """
    Khối NAFBlock (Nonlinear Activation Free) gọn nhẹ:
    Kiến trúc SOTA trong khử mờ ảnh, không dùng hàm kích hoạt nặng nề (ReLU/GELU),
    giúp mô hình nhẹ, hội tụ nhanh và ổn định trên GPU Kaggle / Colab.
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=1)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1)
        
        # Simple Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1)
        )
        self.conv4 = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        # Simple Gate: nhân đôi kênh và nhân với nhau
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv3(x)
        x = x * self.sca(x)
        x = self.conv4(x)
        return res + x

class MoEQFormerDeblurNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feats=64, num_blocks=8, 
                 num_queries=16, num_experts=3):
        """
        Args:
            num_feats (int): Số kênh đặc trưng bên trong mạng.
            num_blocks (int): Số lượng khối NAFBlock của Backbone.
            num_queries (int): Số lượng queries trong Q-Former.
            num_experts (int): Số lượng chuyên gia trong MoE.
        """
        super().__init__()

        # 1. HEAD: Trích xuất đặc trưng sơ cấp nông
        self.head = default_conv(in_channels, num_feats, kernel_size=3)

        # 2. BACKBONE: Chuỗi các khối NAFBlockLite trích xuất đặc trưng sâu
        self.backbone = nn.Sequential(*[NAFBlockLite(num_feats) for _ in range(num_blocks)])

        # 3. BOTTLENECK: Tích hợp Q-Former và MoE
        self.qformer = MiniQFormer(
            in_channels=num_feats, 
            num_queries=num_queries, 
            query_dim=num_feats, 
            num_heads=4, 
            num_layers=2
        )
        self.moe = MoEBlock(
            channels=num_feats, 
            context_dim=num_feats, 
            num_experts=num_experts
        )

        # 4. TAIL: Tái tạo ảnh nét (giữ nguyên kích thước H x W)
        self.tail = nn.Sequential(
            default_conv(num_feats, num_feats, kernel_size=3),
            nn.GELU(),
            default_conv(num_feats, out_channels, kernel_size=3)
        )

    def freeze_backbone(self, freeze=True):
        """
        Đóng băng Backbone để chỉ fine-tune Q-Former + MoE + Head/Tail:
        Rất hữu ích khi chạy trên Kaggle/Colab giúp huấn luyện nhanh gấp nhiều lần.
        """
        for param in self.head.parameters():
            param.requires_grad = not freeze
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        status = "ĐÓNG BĂNG (Freeze)" if freeze else "MỞ KHÓA (Unfreeze)"
        print(f"--> Đã {status} Backbone!")

    def forward(self, x):
        """
        Args:
            x: Ảnh mờ đầu vào (B, 3, H, W)
        Returns:
            out_img: Ảnh phục hồi nét (B, 3, H, W)
            balance_loss: Hàm mất mát cân bằng tải của MoE Router
        """
        # Trích xuất đặc trưng qua Head và Backbone
        feat = self.head(x)
        feat = self.backbone(feat)

        # Bước 1: Q-Former nhận diện bối cảnh mờ và điều biến đặc trưng
        context_vec, modulated_feat = self.qformer(feat)

        # Bước 2: MoE Router phân bổ trọng số cho các Expert
        moe_feat, balance_loss = self.moe(modulated_feat, context_vec)

        # Kết hợp skip connection
        refined_feat = feat + moe_feat

        # Bước 3: Tái tạo ảnh nét với Residual Learning
        res = self.tail(refined_feat)
        out_img = x + res  # Ảnh nét = Ảnh mờ + Phần dư khôi phục

        return out_img, balance_loss

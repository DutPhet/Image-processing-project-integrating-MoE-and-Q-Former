'''
Module Mixture of Experts (MoE) cho Phục hồi ảnh (Image Restoration).
Bao gồm:
- Router: Mạng định tuyến tích hợp thông tin cục bộ và vector bối cảnh từ Q-Former.
- Các Experts chuyên biệt:
    + Smooth/Low-Freq Expert: Phục hồi vùng ảnh phẳng, khử nhòe nền (Large-kernel Conv).
    + Edge/High-Freq Expert: Tái tạo biên cạnh sắc nét (Dilated Conv).
    + Texture Expert: Tái tạo chi tiết phức tạp, hoa văn bề mặt (Gated Conv).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothExpert(nn.Module):
    """Chuyên gia 1: Xử lý vùng phẳng, mịn màng và khử nhiễu nền (dùng Depthwise kernel lớn)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
    def forward(self, x):
        return x + self.conv(x)

class EdgeExpert(nn.Module):
    """Chuyên gia 2: Tái tạo cạnh sắc nhọn và viền đối tượng (dùng Dilated Conv để bắt cấu trúc)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)

class TextureExpert(nn.Module):
    """Chuyên gia 3: Tái tạo hoa văn và chi tiết tần số cao (dùng Gated Convolution)"""
    def __init__(self, channels):
        super().__init__()
        self.conv_feat = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_gate = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        feat = F.gelu(self.conv_feat(x))
        gate = torch.sigmoid(self.conv_gate(x))
        return x + self.proj(feat * gate)

class MoERouter(nn.Module):
    """
    Router định tuyến các chuyên gia:
    Kết hợp giữa đặc trưng không gian cục bộ (Spatial Features) và Context Vector từ Q-Former
    để tính toán trọng số Gating phân bổ cho từng Expert.
    """
    def __init__(self, channels, context_dim=64, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        
        # Mạng tính gating từ pooling của feature map + context vector
        self.gate_net = nn.Sequential(
            nn.Linear(channels + context_dim, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, num_experts)
        )

    def forward(self, x, context_vec=None):
        """
        Args:
            x: Feature map (B, C, H, W)
            context_vec: Vector bối cảnh từ Q-Former (B, context_dim)
        Returns:
            gates: Trọng số phân bổ cho các Expert (B, num_experts, 1, 1)
            balance_loss: Hàm mất mát cân bằng tải (Auxiliary Loss)
        """
        b, c, _, _ = x.shape
        # Global Average Pooling để lấy đặc trưng tổng quát của ảnh
        gap = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        
        if context_vec is not None:
            router_input = torch.cat([gap, context_vec], dim=-1)
        else:
            dummy_ctx = torch.zeros(b, self.gate_net[0].in_features - c, device=x.device)
            router_input = torch.cat([gap, dummy_ctx], dim=-1)
            
        logits = self.gate_net(router_input)  # (B, num_experts)
        gates = F.softmax(logits, dim=-1)      # (B, num_experts)
        
        # Tính Auxiliary Load Balancing Loss để tránh trường hợp Router chỉ gọi duy nhất 1 expert
        # Công thức: loss = num_experts * sum(mean_gate_i^2)
        mean_gates = gates.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(mean_gates ** 2)
        
        # Reshape để nhân trọng số với tensor ảnh: (B, num_experts, 1, 1, 1)
        return gates, balance_loss

class MoEBlock(nn.Module):
    """Khối MoE tổng thể chứa Router và danh sách các Experts"""
    def __init__(self, channels=64, context_dim=64, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.router = MoERouter(channels, context_dim, num_experts=num_experts)
        
        # 3 chuyên gia riêng biệt
        self.experts = nn.ModuleList([
            SmoothExpert(channels),
            EdgeExpert(channels),
            TextureExpert(channels)
        ])

    def forward(self, x, context_vec=None):
        """
        Args:
            x: Tensor (B, C, H, W)
            context_vec: Tensor (B, context_dim) từ Q-Former
        """
        gates, balance_loss = self.router(x, context_vec)
        
        # Chạy từng expert và nhân với trọng số tương ứng
        out = 0
        for i, expert in enumerate(self.experts):
            weight = gates[:, i].view(-1, 1, 1, 1)
            out = out + weight * expert(x)
            
        return out, balance_loss


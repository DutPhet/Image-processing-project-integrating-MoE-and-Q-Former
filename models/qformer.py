'''
Module Q-Former (Querying Transformer) thu nhỏ (Mini Q-Former) cho Phục hồi ảnh (Image Restoration).
Sử dụng các Learnable Queries để trích xuất đặc trưng bối cảnh/loại khuyết tật (blur/degradation context)
từ feature map qua cơ chế Cross-Attention mà không làm mất thông tin không gian của ảnh.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniQFormer(nn.Module):
    def __init__(self, in_channels=64, num_queries=16, query_dim=64, num_heads=4, num_layers=2):
        """
        Args:
            in_channels (int): Số kênh đặc trưng của visual feature map từ backbone.
            num_queries (int): Số lượng truy vấn học được (learnable queries).
            query_dim (int): Chiều vector của mỗi query.
            num_heads (int): Số head trong Multi-Head Attention.
            num_layers (int): Số tầng Transformer trong Q-Former.
        """
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        
        # Learnable Queries (tập vector truy vấn học được)
        self.learnable_queries = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)
        
        # Chiếu feature map visual về chiều query nếu cần
        if in_channels != query_dim:
            self.visual_proj = nn.Conv2d(in_channels, query_dim, kernel_size=1)
        else:
            self.visual_proj = nn.Identity()
            
        self.layers = nn.ModuleList([
            QFormerLayer(query_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Chiếu kết quả context vector về vector điều khiển Router của MoE
        self.context_head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim)
        )

    def forward(self, visual_feat):
        """
        Args:
            visual_feat: Tensor kích thước (B, C, H, W)
        Returns:
            context_vec: Tensor (B, query_dim) - vector bối cảnh điều khiển Router
            modulated_feat: Tensor (B, C, H, W) - đặc trưng ảnh đã được điều biến bối cảnh
        """
        b, c, h, w = visual_feat.shape
        v_feat = self.visual_proj(visual_feat)  # (B, query_dim, H, W)
        
        # Reshape visual features thành visual tokens: (B, H*W, query_dim)
        v_tokens = v_feat.flatten(2).permute(0, 2, 1)
        
        # Mở rộng queries theo batch size: (B, num_queries, query_dim)
        queries = self.learnable_queries.expand(b, -1, -1)
        
        # Cho queries tương tác với visual tokens qua các tầng Q-Former
        for layer in self.layers:
            queries = layer(queries, v_tokens)
            
        # Tổng hợp các query thành 1 vector bối cảnh toàn cục (Global Context Vector)
        # bằng cách average pooling qua các queries
        context_vec = queries.mean(dim=1)  # (B, query_dim)
        context_vec = self.context_head(context_vec)
        
        # Điều biến ngược lại visual_feat bằng FiLM (Feature-wise Linear Modulation)
        # Giúp các đặc trưng ảnh được định hướng bởi bối cảnh mà Q-Former vừa nhận diện
        scale, shift = context_vec.chunk(2, dim=-1) if self.query_dim % 2 == 0 else (context_vec, context_vec)
        # Chuẩn hóa về đúng số kênh c
        if scale.shape[-1] != c:
            scale = F.adaptive_avg_pool1d(scale.unsqueeze(1), c).squeeze(1)
            shift = F.adaptive_avg_pool1d(shift.unsqueeze(1), c).squeeze(1)
            
        scale = scale.view(b, c, 1, 1)
        shift = shift.view(b, c, 1, 1)
        modulated_feat = visual_feat * (1 + torch.tanh(scale)) + shift
        
        return context_vec, modulated_feat

class QFormerLayer(nn.Module):
    """Một tầng gồm Self-Attention (giữa các Queries) và Cross-Attention (Queries với Visual Tokens)"""
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        # 1. Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, visual_tokens):
        # 1. Queries tự học và trao đổi thông tin với nhau
        q_norm = self.norm1(queries)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + attn_out
        
        # 2. Queries trích xuất thông tin từ ảnh (Cross-Attention: Q=queries, K,V=visual_tokens)
        q_norm = self.norm2(queries)
        cross_out, _ = self.cross_attn(query=q_norm, key=visual_tokens, value=visual_tokens)
        queries = queries + cross_out
        
        # 3. Feed Forward
        queries = queries + self.ffn(self.norm3(queries))
        return queries


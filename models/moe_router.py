import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeMoEConv2d(nn.Module):
    """
    MoE-LoRA thay thế lớp Conv2d 1x1 trong FFN của NAFBlock.

    Công thức output:
        out = W0(x)  [frozen base]
            + U · (M_eff · (V_t(x) ⊙ S_diag)) · γ_l  [spectral MoE]
            + lora_B(lora_A(x)) · scale · β_l          [LoRA bypass]

    Trong đó M_eff = σ_l·I + Σ_k gate_k(x, f_global) · Expert_k

    Cải tiến so với phiên bản cũ:
    - Thêm S_diag (singular values từ SVD) vào spectral mixing → đảm bảo
      spec_out ban đầu khớp với pretrained behavior của W0.
    - Bỏ collapse_ratio (gate collapse schedule) — giữ spatial gating ổn
      định trong suốt quá trình training, tránh MoE bị vô hiệu hóa.
    - Clamp balance_loss để tránh gradient explosion.
    - Tăng r=16, num_experts=4 cho capacity tốt hơn.
    """

    def __init__(self, in_channels, out_channels, r=16, rb=32, num_experts=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.num_experts = num_experts

        # ── Base weight W0 (đóng băng, load từ pretrained) ──────────────────
        self.W0 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.W0.weight.requires_grad = False

        # ── SVD anchor: U, V_t, S_diag ──────────────────────────────────────
        # U, V_t là learnable; S_diag là buffer (cố định từ SVD, không train)
        self.U = nn.Parameter(torch.randn(out_channels, r, 1, 1) * 0.02)
        self.V_t = nn.Parameter(torch.randn(r, in_channels, 1, 1) * 0.02)
        # Singular values — dùng register_buffer để lưu/load đúng cách
        # và không bị ảnh hưởng bởi freeze_backbone
        self.register_buffer('S_diag', torch.ones(r))

        # ── Learnable scaling scalars ────────────────────────────────────────
        self.gamma_l = nn.Parameter(torch.tensor(0.1))   # spectral scale
        self.sigma_l = nn.Parameter(torch.tensor(0.05))  # identity regularizer
        self.beta_l  = nn.Parameter(torch.tensor(1.0))   # LoRA scale

        # ── Expert matrices M_k ─────────────────────────────────────────────
        # Khởi tạo zeros → ban đầu M_eff = σ_l·I (identity-like)
        self.experts = nn.Parameter(torch.zeros(num_experts, r, r))

        # ── LoRA bypass (A: down-project, B: up-project) ────────────────────
        self.lora_A = nn.Conv2d(in_channels, rb, 1, bias=False)
        self.lora_B = nn.Conv2d(rb, out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_scale = 32.0 / rb

        # ── Spatial Router: f(x, f_global) → gates per pixel ────────────────
        # Input: cat(x, f_global_expanded) → 2 * in_channels
        self.router = nn.Conv2d(2 * in_channels, num_experts, kernel_size=1)
        # Learnable global bias gate (fallback khi routing không chắc chắn)
        self.m_g = nn.Parameter(torch.zeros(1, num_experts, 1, 1))

    def init_svd(self, w_pretrained):
        """
        Khởi tạo U, V_t, S_diag từ SVD của pretrained weight W0.
        Đảm bảo spec_out(t=0) ≈ 0 (lora_B=zeros) và W0 được giữ nguyên.
        """
        with torch.no_grad():
            self.W0.weight.copy_(w_pretrained)
            w_mat = w_pretrained.squeeze()
            U, S, Vh = torch.linalg.svd(w_mat, full_matrices=False)
            r = self.r

            self.U.data.copy_(U[:, :r].unsqueeze(-1).unsqueeze(-1))
            self.V_t.data.copy_(Vh[:r, :].unsqueeze(-1).unsqueeze(-1))
            # Lưu singular values vào buffer
            self.S_diag.copy_(S[:r])

    def forward(self, x, f_global):
        """
        Args:
            x:        (B, C_in, H, W)  — feature map đầu vào
            f_global: (B, C_in, 1, 1)  — đặc trưng suy thoái toàn cục

        Returns:
            out:          (B, C_out, H, W)
            balance_loss: scalar — auxiliary load-balancing loss
        """
        b, c, h, w = x.shape

        # ── 1. Base output (frozen pretrained) ──────────────────────────────
        base_out = self.W0(x)

        # ── 2. LoRA bypass ───────────────────────────────────────────────────
        byp_out = self.lora_B(self.lora_A(x)) * self.lora_scale * self.beta_l

        # ── 3. Spatial Router ────────────────────────────────────────────────
        f_global_expanded = f_global.expand(-1, -1, h, w)
        router_input = torch.cat([x, f_global_expanded], dim=1)
        # Kết hợp spatial gating và learned global bias
        z = self.router(router_input) + self.m_g  # (B, E, H, W)
        gates = F.softmax(z, dim=1)               # (B, E, H, W)

        # Balance loss: khuyến khích các expert được dùng đều nhau
        # Clamp để tránh gradient explosion (vấn đề khi aux_weight lớn)
        balance_loss = self.num_experts * (gates.mean(dim=(0, 2, 3)) ** 2).sum()
        balance_loss = torch.clamp(balance_loss, max=1.0)

        # ── 4. Spectral MoE mixing ───────────────────────────────────────────
        # Project x vào không gian low-rank r
        v_x = F.conv2d(x, self.V_t.to(x.dtype))          # (B, r, H, W)
        # Scale bởi singular values → giữ magnitude đúng với pretrained
        s_scale = self.S_diag.to(x.dtype).view(1, self.r, 1, 1)
        sv_x = v_x * s_scale                              # (B, r, H, W)

        # Tính weighted sum của expert matrices theo gate
        gates_flat    = gates.permute(0, 2, 3, 1).reshape(-1, self.num_experts)
        experts_flat  = self.experts.view(self.num_experts, -1)
        mixed_M       = torch.matmul(gates_flat, experts_flat)
        mixed_M       = mixed_M.view(b, h, w, self.r, self.r)

        # Regularization: M_eff = σ_l·I + mixed_M
        I      = torch.eye(self.r, device=x.device, dtype=x.dtype).view(1, 1, 1, self.r, self.r)
        M_eff  = self.sigma_l * I + mixed_M               # (B, H, W, r, r)

        # Apply M_eff to sv_x per pixel
        sv_x_re = sv_x.permute(0, 2, 3, 1).unsqueeze(-1) # (B, H, W, r, 1)
        m_out   = torch.matmul(M_eff, sv_x_re)            # (B, H, W, r, 1)
        m_out   = m_out.squeeze(-1).permute(0, 3, 1, 2)   # (B, r, H, W)

        # Project back lên C_out
        spec_out = F.conv2d(m_out, self.U.to(x.dtype)) * self.gamma_l  # (B, C_out, H, W)

        return base_out + spec_out + byp_out, balance_loss

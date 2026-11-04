import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeMoEConv2d(nn.Module):
    """
    SVD-Anchored Spectral MoE + LoRA Bypass với Zero-Initialization.
    """
    def __init__(self, in_channels, out_channels, r=16, rb=32, num_experts=4, top_k=2, global_ch=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        global_ch = global_ch if global_ch is not None else in_channels

        # ── 1. Base weight W0 (Pretrained Stage 1) ──────────────────────────
        self.W0 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.W0.weight.requires_grad = False

        # ── 2. SVD anchors ──────────────────────────────────────────────────
        self.U = nn.Parameter(torch.randn(out_channels, r, 1, 1) * 0.02)
        self.V_t = nn.Parameter(torch.randn(r, in_channels, 1, 1) * 0.02)
        self.register_buffer('sigma_1_0', torch.tensor(1.0))

        # ── 3. Zero-Initialization Scaling Parameters ───────────────────────
        self.gamma_l = nn.Parameter(torch.tensor(0.0)) 
        self.sigma_l = nn.Parameter(torch.tensor(0.05))
        self.beta_l = nn.Parameter(torch.tensor(0.0))

        # ── 4. Expert matrices M_k ──────────────────────────────────────────
        self.experts = nn.Parameter(torch.zeros(num_experts, r, r))

        # ── 5. LoRA bypass ──────────────────────────────────────────────────
        self.lora_A = nn.Conv2d(in_channels, rb, 1, bias=False)
        self.lora_B = nn.Conv2d(rb, out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_scale = 32.0 / rb

        # ── 6. Spatial Router ───────────────────────────────────────────────
        self.f_proj = nn.Conv2d(global_ch, in_channels, 1) if global_ch != in_channels else nn.Identity()
        self.router = nn.Conv2d(2 * in_channels, num_experts, kernel_size=1)
        self.m_g = nn.Parameter(torch.zeros(1, num_experts, 1, 1))

    def init_svd(self, w_pretrained):
        with torch.no_grad():
            self.W0.weight.copy_(w_pretrained)
            w_mat = w_pretrained.view(self.out_channels, self.in_channels)
            U, S, Vh = torch.linalg.svd(w_mat, full_matrices=False)
            r_eff = min(self.r, min(self.out_channels, self.in_channels))

            self.U.zero_()
            self.V_t.zero_()
            self.U.data[:, :r_eff, 0, 0].copy_(U[:, :r_eff])
            self.V_t.data[:r_eff, :, 0, 0].copy_(Vh[:r_eff, :])

            if S.numel() > 0:
                self.sigma_1_0.copy_(S[0])

    def forward(self, x, f_global):
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        base_out = self.W0(x)
        byp_out = self.lora_B(self.lora_A(x)) * self.lora_scale * self.beta_l.to(device=device, dtype=dtype)

        f_global_mapped = self.f_proj(f_global)
        f_global_expanded = f_global_mapped.expand(-1, -1, h, w)

        router_input = torch.cat([x, f_global_expanded], dim=1)
        z = self.router(router_input) + self.m_g

        topk_val, topk_idx = torch.topk(z, self.top_k, dim=1)
        topk_gates = F.softmax(topk_val, dim=1)

        gates = torch.zeros_like(z).scatter_(1, topk_idx, topk_gates)

        # ── Load Balance Loss chuẩn Switch-Transformer ─────────────────────
        P_k = F.softmax(z, dim=1).mean(dim=(0, 2, 3))               # Soft routing prob
        f_k = (gates > 0).float().mean(dim=(0, 2, 3))                # Hard selection fraction
        balance_loss = self.num_experts * torch.sum(f_k * P_k)

        v_x = F.conv2d(x, self.V_t.to(device=device, dtype=dtype))
        v_x_flat = v_x.flatten(2)
        m_all_flat = torch.einsum('k r j, b j n -> b k r n', self.experts.to(device=device, dtype=dtype), v_x_flat)

        gates_flat = gates.flatten(2)
        moe_term_flat = torch.einsum('b k n, b k r n -> b r n', gates_flat, m_all_flat)
        moe_term = moe_term_flat.view(b, self.r, h, w)

        m_out = torch.abs(self.sigma_l.to(device=device, dtype=dtype)) * v_x + moe_term
        spec_scale = self.gamma_l.to(device=device, dtype=dtype) * self.sigma_1_0.to(device=device, dtype=dtype)
        spec_out = F.conv2d(m_out, self.U.to(device=device, dtype=dtype)) * spec_scale

        return base_out + spec_out + byp_out, balance_loss
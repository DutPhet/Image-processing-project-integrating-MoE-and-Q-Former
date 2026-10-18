'''
Mô hình chuẩn U-Net cho NAFNet Baseline và MoEQFormerDeblurNet:
- NAFNetBaseline: Mô hình thuần túy NAFNet U-Net khớp 100% trọng số checkpoint gốc.
- MoEQFormerDeblurNet: Mô hình lai tích hợp MiniQFormer và MoE Block vào tầng cổ chai (Bottleneck / middle_blks) 
  của cùng kiến trúc U-Net, đảm bảo so sánh công bằng và khoa học.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import default_conv
from .qformer import MiniQFormer
from .moe_router import MoEBlock

class LayerNorm2d(nn.Module):
    """Layer Normalization cho Tensor 2D (B, C, H, W) chuẩn của NAFNet"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """Khối NAFBlock chuẩn tương thích 100% với Pretrained NAFNet-GoPro-width64.pth"""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )
        
        ffn_channel = c * FFN_Expand
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sg = SimpleGate()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNetBaseline(nn.Module):
    """
    1. Mô hình NAFNet gốc thuần túy dạng U-Net (Khớp 100% checkpoint gốc với strict=True)
    Dùng để đo baseline chuẩn paper (~30-32 dB).
    """
    def __init__(self, img_channel=3, width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, x):
        inp = x
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        out_img = x + inp
        return out_img, 0.0


class MoEQFormerDeblurNet(nn.Module):
    """
    2. Mô hình lai tích hợp Q-Former và MoE trên nền tảng U-Net chuẩn của NAFNet:
    - Giữ nguyên Encoder, Decoder và các khối Skip Connection giúp nạp tối đa pretrained weights.
    - Tích hợp Q-Former và MoE tại tầng cổ chai (middle_blks).
    """
    def __init__(self, img_channel=3, out_channels=3, width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2],
                 num_queries=16, num_experts=3):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        # Cổ chai gồm các khối NAFBlock
        self.middle_blks = nn.ModuleList([NAFBlock(chan) for _ in range(middle_blk_num)])

        # Tích hợp Q-Former và MoE tại cổ chai
        self.qformer = MiniQFormer(
            in_channels=chan, 
            num_queries=num_queries, 
            query_dim=chan, 
            num_heads=4, 
            num_layers=2
        )
        self.moe = MoEBlock(
            channels=chan, 
            context_dim=chan, 
            num_experts=num_experts
        )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def freeze_backbone(self, freeze=True):
        """Đóng băng các tầng U-Net gốc để chỉ fine-tune Q-Former và MoE"""
        for m in [self.intro, self.ending, self.encoders, self.decoders, self.middle_blks, self.ups, self.downs]:
            for param in m.parameters():
                param.requires_grad = not freeze
        status = "ĐÓNG BĂNG (Freeze)" if freeze else "MỞ KHÓA (Unfreeze)"
        print(f"--> Đã {status} toàn bộ U-Net Backbone!")

    def forward(self, x):
        inp = x
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Chạy qua middle blocks
        for blk in self.middle_blks:
            x = blk(x)

        # Áp dụng Q-Former và MoE tại cổ chai
        context_vec, modulated_feat = self.qformer(x)
        moe_feat, balance_loss = self.moe(modulated_feat, context_vec)
        x = x + moe_feat

        # Qua Decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        out_img = x + inp
        return out_img, balance_loss
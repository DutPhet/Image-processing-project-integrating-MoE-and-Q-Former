import torch
import torch.nn as nn
from .common import default_conv
from .moe_router import ForgeMoEConv2d


class LayerNorm2d(nn.Module):
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
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1)
        )
        ffn_channel = c * FFN_Expand
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sg = SimpleGate()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

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


class FORGENAFBlock(nn.Module):
    """
    NAFBlock với FFN projection được thay thế bởi ForgeMoEConv2d.
    Cấy ghép MoE-LoRA vào conv4, conv5 với điều kiện đặc trưng suy thoái toàn cục f_global.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., r=16, rb=32, num_experts=4):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1)
        )

        ffn_channel = c * FFN_Expand
        self.conv4 = ForgeMoEConv2d(c, ffn_channel, r=r, rb=rb, num_experts=num_experts)
        self.conv5 = ForgeMoEConv2d(ffn_channel // 2, c, r=r, rb=rb, num_experts=num_experts)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sg = SimpleGate()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp, f_global):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x_norm = self.norm2(y)
        x, loss1 = self.conv4(x_norm, f_global)
        x = self.sg(x)
        x, loss2 = self.conv5(x, f_global)

        x = self.dropout2(x)
        return y + x * self.gamma, loss1 + loss2


class NAFNetBaseline(nn.Module):
    """NAFNet gốc dùng để đánh giá chuẩn baseline."""
    def __init__(self, img_channel=3, width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.ModuleList([NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
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

        for blk in self.middle_blks:
            x = blk(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        return x + inp, 0.0


class FORGEDeblurNet(nn.Module):
    """
    FORGE Deblurring Network tích hợp MoE-LoRA đa vị trí:
    1. Cấy MoE ở Bottleneck (`middle_blks`): Học điều chỉnh đặc trưng sâu, toàn cục.
    2. Cấy MoE ở Decoder cuối (`decoders[-1]`): Độ phân giải lớn nhất (1x), trực tiếp
       phục hồi các chi tiết tần số cao và biên cạnh nét.
    3. Mở khóa linh hoạt (PEFT Unfreezing): Cho phép mở khóa một phần Decoder/Upsampling
       để giải mã tối đa tri thức từ các chuyên gia MoE.
    """
    def __init__(
        self,
        img_channel=3,
        out_channels=3,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        num_queries=16,
        num_experts=4,
        r=16,
        rb=32,
        moe_in_decoder=True,
        decoder_r=8,
        decoder_rb=16,
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.moe_in_decoder = moe_in_decoder

        # Module trích xuất đặc trưng thoái hóa toàn cục (2 lớp Conv + GELU + Pool)
        global_feat_ch = width * 2  # 128 kênh
        self.global_extractor = nn.Sequential(
            nn.Conv2d(img_channel, global_feat_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(global_feat_ch, global_feat_ch, 3, 1, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        # Middle bottleneck với MoE
        self.mid_f_mapper = nn.Conv2d(global_feat_ch, chan, 1)
        self.middle_blks = nn.ModuleList([
            FORGENAFBlock(chan, num_experts=num_experts, r=r, rb=rb)
            for _ in range(middle_blk_num)
        ])

        # Decoders
        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2

            # Cấy MoE vào khối cuối cùng của Decoder (độ phân giải 1x, quyết định độ sắc nét)
            if self.moe_in_decoder and (idx == len(dec_blk_nums) - 1):
                self.decoders.append(nn.ModuleList([
                    FORGENAFBlock(chan, num_experts=num_experts, r=decoder_r, rb=decoder_rb)
                    for _ in range(num)
                ]))
                self.dec_f_mapper = nn.Conv2d(global_feat_ch, chan, 1)
            else:
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def freeze_backbone(self, freeze=True, unfreeze_decoders=True, unfreeze_last_dec=True):
        """
        Chiến lược mở khóa thông minh (PEFT):
        - unfreeze_decoders=True: Mở TOÀN BỘ Decoder + Upsampling + Ending conv để giải mã MoE.
        - unfreeze_last_dec=True: Chỉ mở khối decoder cuối cùng (1x) + Ending conv.
        """
        for param in self.parameters():
            param.requires_grad = not freeze

        if not freeze:
            return

        # 1. Luôn mở Extractor & Feature Mappers
        for param in self.global_extractor.parameters():
            param.requires_grad = True
        for param in self.mid_f_mapper.parameters():
            param.requires_grad = True
        if self.moe_in_decoder:
            for param in self.dec_f_mapper.parameters():
                param.requires_grad = True

        # 2. Luôn mở các tham số MoE ở middle
        for blk in self.middle_blks:
            for moe in [blk.conv4, blk.conv5]:
                for name, param in moe.named_parameters():
                    if name != 'W0.weight':
                        param.requires_grad = True

        # 3. Luôn mở các tham số MoE ở decoder cuối
        if self.moe_in_decoder:
            for blk in self.decoders[-1]:
                if isinstance(blk, FORGENAFBlock):
                    for moe in [blk.conv4, blk.conv5]:
                        for name, param in moe.named_parameters():
                            if name != 'W0.weight':
                                param.requires_grad = True

        # 4. Mở khóa linh hoạt theo kỹ thuật PEFT (Ý TƯỞNG B)
        if unfreeze_decoders:
            for param in self.decoders.parameters():
                param.requires_grad = True
            for param in self.ups.parameters():
                param.requires_grad = True
            for param in self.ending.parameters():
                param.requires_grad = True
        elif unfreeze_last_dec:
            for param in self.decoders[-1].parameters():
                param.requires_grad = True
            for param in self.ending.parameters():
                param.requires_grad = True

        # Đóng băng tuyệt đối trọng số W0 của ForgeMoEConv2d để bảo toàn tri thức nền
        for m in self.modules():
            if isinstance(m, ForgeMoEConv2d):
                m.W0.weight.requires_grad = False

    def load_pretrained(self, state_dict):
        """
        Nạp trọng số NAFNet pretrained và cấy ghép SVD-Anchor cho các khối MoE.
        """
        new_state_dict = {}
        last_dec_idx = len(self.decoders) - 1

        for k, v in state_dict.items():
            if 'middle_blks' in k and ('conv4.weight' in k or 'conv5.weight' in k):
                new_k = k.replace('.weight', '.W0.weight')
                new_state_dict[new_k] = v
            elif self.moe_in_decoder and f'decoders.{last_dec_idx}' in k and ('conv4.weight' in k or 'conv5.weight' in k):
                new_k = k.replace('.weight', '.W0.weight')
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)

        # Khởi tạo SVD anchor cho tất cả ForgeMoEConv2d (ở middle và decoder)
        for m in self.modules():
            if isinstance(m, ForgeMoEConv2d):
                m.init_svd(m.W0.weight)
        print("--> Đã nạp thành công Checkpoint và cấy ghép SVD-Anchor FORGE MoE (Middle + Decoder).")

    def forward(self, x):
        inp = x

        # Trích xuất đặc trưng thoái hóa toàn cục
        f_global_base = self.global_extractor(x)

        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        total_balance_loss = 0.0

        # Bottleneck với MoE
        f_global_mid = self.mid_f_mapper(f_global_base)
        for blk in self.middle_blks:
            x, blk_loss = blk(x, f_global_mid)
            total_balance_loss = total_balance_loss + blk_loss

        # Decoders
        last_dec_idx = len(self.decoders) - 1
        for idx, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip

            if self.moe_in_decoder and (idx == last_dec_idx):
                f_global_dec = self.dec_f_mapper(f_global_base)
                for blk in decoder:
                    x, dec_loss = blk(x, f_global_dec)
                    total_balance_loss = total_balance_loss + dec_loss
            else:
                x = decoder(x)

        x = self.ending(x)
        return x + inp, total_balance_loss
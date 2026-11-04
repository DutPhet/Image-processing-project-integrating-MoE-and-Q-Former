import torch
import torch.nn as nn
import torch.nn.functional as F
from .forge import LayerNorm2d, SimpleGate, NAFBlock
from .moe_router import ForgeMoEConv2d


class DualPoolGlobalExtractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch // 2, 3, 1, 1),
            nn.GELU(),
        )
        self.fc = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        feat = self.conv(x)
        avg_p = F.adaptive_avg_pool2d(feat, 1)
        std_p = torch.std(feat, dim=(-2, -1), keepdim=True)
        out = torch.cat([avg_p, std_p], dim=1)
        return self.fc(out)


class MultiGranularityPGM(nn.Module):
    def __init__(self, prompt_dim=128, num_prompts=10, key_dim=32, num_tasks=6, num_granularity=3):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.num_tasks = num_tasks
        self.num_granularity = num_granularity

        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, key_dim))
        self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))

        self.query_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(prompt_dim, key_dim)
        )

        self.task_head = nn.Linear(prompt_dim, num_tasks)
        self.granularity_head = nn.Linear(prompt_dim, num_granularity)

        self.fusion_proj = nn.Sequential(
            nn.Linear(num_tasks + num_granularity, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim)
        )

        nn.init.orthogonal_(self.prompt_keys)
        nn.init.normal_(self.prompt_values, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query_proj(x)

        sim = torch.matmul(q, self.prompt_keys.T) / (self.prompt_keys.shape[1] ** 0.5)
        weights = F.softmax(sim, dim=-1)
        prompt = torch.matmul(weights, self.prompt_values)

        task_logits = self.task_head(prompt)
        granularity_logits = self.granularity_head(prompt)

        task_probs = F.softmax(task_logits, dim=-1)
        gran_probs = F.softmax(granularity_logits, dim=-1)

        joint_feat = torch.cat([task_probs, gran_probs], dim=-1)
        routing_context = self.fusion_proj(joint_feat).view(b, c, 1, 1)

        prompt_feat = prompt.view(b, c, 1, 1) + routing_context
        return prompt_feat, task_logits, granularity_logits


class PromptFORGENAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., r=16, rb=32, num_experts=4, top_k=2):
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
        self.conv4 = ForgeMoEConv2d(c, ffn_channel, r=r, rb=rb, num_experts=num_experts, top_k=top_k, global_ch=c)
        self.conv5 = ForgeMoEConv2d(ffn_channel // 2, c, r=r, rb=rb, num_experts=num_experts, top_k=top_k, global_ch=c)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sg = SimpleGate()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma_prompt = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp, prompt_feat):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x_norm = self.norm2(y) + self.gamma_prompt * prompt_feat
        
        x, loss1 = self.conv4(x_norm, prompt_feat)
        x = self.sg(x)
        x, loss2 = self.conv5(x, prompt_feat)

        x = self.dropout2(x)
        return y + x * self.gamma, loss1 + loss2


class PromptFORGEDeblurNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        out_channels=3,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        num_prompts=10,
        num_experts=4,
        top_k=2,
        r=16,
        rb=32,
        moe_in_decoder=True,
        decoder_r=8,
        decoder_rb=16,
        num_tasks=6,
        num_granularity=3
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.moe_in_decoder = moe_in_decoder

        global_feat_ch = width * 2
        self.global_extractor = DualPoolGlobalExtractor(img_channel, global_feat_ch)
        self.pgm = MultiGranularityPGM(
            prompt_dim=global_feat_ch, 
            num_prompts=num_prompts, 
            num_tasks=num_tasks,
            num_granularity=num_granularity
        )

        self.enc_f_mappers = nn.ModuleList()
        chan = width
        for idx, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            setattr(self, f'enc_prompt_scale_{idx}', nn.Parameter(torch.zeros(1, chan, 1, 1)))
            
            # ── ĐÃ SỬA: Mapper ánh xạ global_feat_ch sang chan của từng Encoder stage ────
            mapper = nn.Conv2d(global_feat_ch, chan, 1)
            nn.init.zeros_(mapper.weight)
            nn.init.zeros_(mapper.bias)
            self.enc_f_mappers.append(mapper)
            
            chan = chan * 2

        self.mid_f_mapper = nn.Conv2d(global_feat_ch, chan, 1)
        nn.init.zeros_(self.mid_f_mapper.weight)
        nn.init.zeros_(self.mid_f_mapper.bias)

        self.middle_blks = nn.ModuleList([
            PromptFORGENAFBlock(chan, num_experts=num_experts, top_k=top_k, r=r, rb=rb)
            for _ in range(middle_blk_num)
        ])

        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2

            if self.moe_in_decoder and (idx == len(dec_blk_nums) - 1):
                self.decoders.append(nn.ModuleList([
                    PromptFORGENAFBlock(chan, num_experts=num_experts, top_k=top_k, r=decoder_r, rb=decoder_rb)
                    for _ in range(num)
                ]))
                self.dec_f_mapper = nn.Conv2d(global_feat_ch, chan, 1)
                nn.init.zeros_(self.dec_f_mapper.weight)
                nn.init.zeros_(self.dec_f_mapper.bias)
            else:
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def freeze_backbone(self, freeze=True, unfreeze_last_dec_only=False):
        if not freeze:
            for param in self.parameters():
                param.requires_grad = True
            return

        for param in self.parameters():
            param.requires_grad = False

        for param in self.global_extractor.parameters():
            param.requires_grad = True
        for param in self.pgm.parameters():
            param.requires_grad = True
        for mapper in self.enc_f_mappers:
            for param in mapper.parameters():
                param.requires_grad = True
        for param in self.mid_f_mapper.parameters():
            param.requires_grad = True
            
        if self.moe_in_decoder:
            for param in self.dec_f_mapper.parameters():
                param.requires_grad = True

        for m in self.modules():
            if isinstance(m, ForgeMoEConv2d):
                for name, param in m.named_parameters():
                    if name != 'W0.weight':
                        param.requires_grad = True

        if unfreeze_last_dec_only:
            for param in self.decoders[-1].parameters():
                param.requires_grad = True
            for param in self.ending.parameters():
                param.requires_grad = True

        for m in self.modules():
            if isinstance(m, ForgeMoEConv2d):
                m.W0.weight.requires_grad = False

    def forward(self, x):
        inp = x

        feat_base = self.global_extractor(x)
        prompt_base, task_logits, granularity_logits = self.pgm(feat_base)

        x = self.intro(x)

        encs = []
        for idx, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x)
            scale = getattr(self, f'enc_prompt_scale_{idx}')
            # ── ĐÃ SỬA: Chiếu Prompt về đúng kênh trước khi nhân Skip-Connection ──
            prompt_enc = self.enc_f_mappers[idx](prompt_base)
            encs.append(x * (1.0 + scale * prompt_enc))
            x = down(x)

        total_balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        prompt_mid = self.mid_f_mapper(prompt_base)
        for blk in self.middle_blks:
            x, blk_loss = blk(x, prompt_mid)
            total_balance_loss = total_balance_loss + blk_loss

        last_dec_idx = len(self.decoders) - 1
        for idx, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip

            if self.moe_in_decoder and (idx == last_dec_idx):
                prompt_dec = self.dec_f_mapper(prompt_base)
                for blk in decoder:
                    x, dec_loss = blk(x, prompt_dec)
                    total_balance_loss = total_balance_loss + dec_loss
            else:
                x = decoder(x)

        x = self.ending(x)
        return x + inp, total_balance_loss, task_logits, granularity_logits
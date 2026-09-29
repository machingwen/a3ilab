"""Three-condition Add U-Net, trainer, and DDIM sampler for CCDM.

Condition order is (hair style, hair color, sex). The U-Net blocks and
two-condition embedding definitions come from the original CCDM code.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.embedding import ConditionalEmbedding
from models.engine import extract
from models.unet import UNet


class TripleUNet(UNet):
    def __init__(self, num_style=2, num_color=4, num_sex=2, compose=False,
                 T=1000, model_channels=128, ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2, dropout=0.15, drop_prob=0.1):
        # The parent creates the identical two-condition U-Net blocks. Its
        # condition embeddings are reused for style and color.
        super().__init__(T=T, num_atr=num_style, num_obj=num_color,
                         model_channels=model_channels, ch_mult=ch_mult,
                         num_res_blocks=num_res_blocks, dropout=dropout,
                         drop_prob=drop_prob, compose=False)
        tdim = model_channels * 4
        self.sex_embedding = ConditionalEmbedding(num_sex, model_channels,
                                                  tdim, drop_prob)
        self.compose = compose
        if compose:
            # Exact two-input CCDM projection pattern, extended to three.
            self.projection = nn.Sequential(
                nn.Linear(tdim * 3, 4096), nn.LayerNorm(4096),
                nn.Dropout(0.5), nn.Linear(4096, tdim))

    def forward(self, x, t, style, color, sex, force_drop_ids=None):
        style_emb = self.atr_embedding(style, force_drop_ids=force_drop_ids)
        color_emb = self.obj_embedding(color, force_drop_ids=force_drop_ids)
        sex_emb = self.sex_embedding(sex, force_drop_ids=force_drop_ids)
        if self.compose:
            cond = self.projection(torch.cat((style_emb, color_emb, sex_emb), dim=-1))
        else:
            cond = style_emb + color_emb + sex_emb
        temb = self.time_embedding(t) + cond
        hs = []
        h = x
        for layer in self.input_blocks:
            h = layer(h, temb)
            hs.append(h)
        h = self.middle_block(h, temb)
        for layer in self.output_blocks:
            h = torch.cat((h, hs.pop()), dim=1)
            h = layer(h, temb)
        return self.out(h)


class TripleDiffusionTrainer(nn.Module):
    def __init__(self, model, beta=(0.0001, 0.02), T=1000):
        super().__init__()
        self.model = model
        self.T = T
        betas = torch.linspace(*beta, T, dtype=torch.float32)
        abar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("signal_rate", torch.sqrt(abar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - abar))

    def forward(self, x, style, color, sex):
        t = torch.randint(self.T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)
        xt = extract(self.signal_rate, t, x.shape) * x + extract(self.noise_rate, t, x.shape) * eps
        return F.mse_loss(self.model(xt, t, style, color, sex), eps, reduction="none")


class TripleDDIMSampler(nn.Module):
    def __init__(self, model, beta=(0.0001, 0.02), T=1000, w=1.8):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w
        abar = torch.cumprod(1.0 - torch.linspace(*beta, T, dtype=torch.float32), dim=0)
        self.register_buffer("alpha_t_bar", abar)

    @torch.no_grad()
    def forward(self, x, style, color, sex, steps=100, eta=0.0):
        if not (1 <= steps < self.T and self.T % steps == 0):
            raise ValueError("steps must divide T and be less than T")
        # Preserve the archived CCDM DDIM grid, including its +1 offset.
        times = np.arange(0, self.T, self.T // steps, dtype=np.int64) + 1
        previous = np.concatenate(([0], times[:-1]))
        for t_i, prev_i in zip(times[::-1], previous[::-1]):
            t = torch.full((x.shape[0],), int(t_i), device=x.device, dtype=torch.long)
            p = torch.full_like(t, int(prev_i))
            a = extract(self.alpha_t_bar, t, x.shape)
            ap = extract(self.alpha_t_bar, p, x.shape)
            conditional = self.model(x, t, style, color, sex)
            unconditional = self.model(x, t, style, color, sex, force_drop_ids=True)
            eps = (1 + self.w) * conditional - self.w * unconditional
            sigma = eta * torch.sqrt((1 - ap) / (1 - a) * (1 - a / ap))
            noise = torch.randn_like(x) if eta else 0
            x = (torch.sqrt(ap / a) * x
                 + (torch.sqrt(1 - ap - sigma.square()) - torch.sqrt(ap * (1 - a) / a)) * eps
                 + sigma * noise)
        return x

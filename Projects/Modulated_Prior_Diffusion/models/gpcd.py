import torch
from torch import nn


class GPCDConcatDownsizer(nn.Module):
    def __init__(self, denoiser: nn.Module, in_channels: int = 3):
        super().__init__()
        self.denoiser = denoiser
        self._printed_debug_shapes = False
        self.downsizer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

    def forward(self, x, timesteps=None, c2=None, **kwargs):
        if c2 is None:
            raise ValueError("gpcd_concat requires reference image c2")
        if not self._printed_debug_shapes:
            print(f"GPCDConcatDownsizer forward: x.shape={tuple(x.shape)} c2.shape={tuple(c2.shape)}")
            self._printed_debug_shapes = True
        x = self.downsizer(torch.cat([x, c2], dim=1))
        return self.denoiser(x, timesteps, **kwargs)

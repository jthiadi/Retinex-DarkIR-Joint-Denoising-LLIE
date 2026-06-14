"""Lightweight illumination estimator for Retinex-DarkIR."""

import torch
import torch.nn as nn


class IlluminationBranch(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1, groups=base),
            nn.GELU(),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).clamp(min=0.05, max=1.0)

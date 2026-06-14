"""Validation metrics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        return float("nan")
    mse = F.mse_loss(pred, target).item()
    if not math.isfinite(mse) or mse < eps:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    from losses import ssim_map

    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        return float("nan")
    return float(ssim_map(pred, target).mean().item())


class LPIPSMetric(nn.Module):
    """VGG LPIPS (lower is better). Weights downloaded on first use."""

    def __init__(self, net: str = "vgg", device: torch.device | None = None):
        super().__init__()
        import lpips

        self.model = lpips.LPIPS(net=net, verbose=False)
        if device is not None:
            self.model = self.model.to(device)
        self.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            return float("nan")
        # lpips expects NCHW in [-1, 1]
        p = pred.float() * 2.0 - 1.0
        t = target.float() * 2.0 - 1.0
        return float(self.model(p, t).mean().item())

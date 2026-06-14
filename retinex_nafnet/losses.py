from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def hetero_weights(low: torch.Tensor) -> torch.Tensor:
    lum = rgb_to_luma(low)
    return 1.0 + (1.0 - lum).pow(2)


def downsample_for_loss(x: torch.Tensor, max_side: int) -> torch.Tensor:
    """Downsample for MS-SSIM / FFT to cut VRAM on large crops (e.g. 512 on 6 GB GPUs)."""
    if max_side <= 0:
        return x
    h, w = x.shape[-2:]
    m = max(h, w)
    if m <= max_side:
        return x
    scale = max_side / m
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)


def _gaussian_window(channels: int, window: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=torch.float32) - window // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    w2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w2d.expand(channels, 1, window, window).contiguous()


def ssim_map(
    pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    c = pred.shape[1]
    w = _gaussian_window(c, window, sigma, pred.device)
    mu_x = F.conv2d(pred, w, padding=window // 2, groups=c)
    mu_y = F.conv2d(target, w, padding=window // 2, groups=c)
    sigma_x = F.conv2d(pred * pred, w, padding=window // 2, groups=c) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, w, padding=window // 2, groups=c) - mu_y * mu_y
    sigma_xy = F.conv2d(pred * target, w, padding=window // 2, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return num / den.clamp(min=1e-8)


def msssim(pred: torch.Tensor, target: torch.Tensor, levels: int = 5) -> torch.Tensor:
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=pred.device)
    weights = weights[:levels]
    weights = weights / weights.sum()
    val = pred.new_tensor(1.0)
    for i in range(levels):
        if i > 0:
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)
        s = ssim_map(pred, target).mean(dim=(1, 2, 3))
        # SSIM can be slightly negative; s ** fractional_weight -> NaN. Clamp for stability.
        s = s.clamp(min=1e-6, max=1.0)
        val = val * s ** weights[i]
    return val.mean()


def fft_mag(x: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_luma(x)
    f = torch.fft.rfft2(gray, norm="ortho")
    return torch.log1p(f.abs())


def _vgg19_weights():
    w = models.VGG19_Weights
    return getattr(w, "IMAGENET1K_FEATURES", w.IMAGENET1K_V1)


class VGGPerceptual(nn.Module):
    def __init__(self, max_size: int = 224):
        super().__init__()
        self.max_size = max_size
        vgg = models.vgg19(weights=_vgg19_weights()).features
        self.slice = nn.Sequential(*list(vgg.children())[:12]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = max(h, w)
        if m <= self.max_size:
            return x
        scale = self.max_size / m
        nh, nw = int(h * scale), int(w * scale)
        return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._resize(pred)
        target = self._resize(target)
        p = (pred - self.mean) / self.std
        t = (target - self.mean) / self.std
        return F.l1_loss(self.slice(p), self.slice(t))


class RestorationLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.w_char = cfg.get("w_char", 1.0)
        self.w_l1 = cfg.get("w_l1", 0.5)
        self.w_luma = cfg.get("w_luma", 0.5)
        self.w_msssim = cfg.get("w_msssim", 0.2)
        self.w_perc = cfg.get("w_perc", 0.08)
        self.w_fft = cfg.get("w_fft", 0.05)
        self.hetero = cfg.get("hetero", True)
        self.loss_max_side = int(cfg.get("loss_max_side", 256))
        self.msssim_levels = int(cfg.get("msssim_levels", 5))
        perc_size = int(cfg.get("perc_size", 224))
        self.perceptual = VGGPerceptual(max_size=perc_size) if self.w_perc > 0 else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, low: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        w = hetero_weights(low) if self.hetero else 1.0
        diff = pred - target

        char = (w * charbonnier(diff)).mean()
        l1 = (w * diff.abs()).mean()
        luma = F.l1_loss(rgb_to_luma(pred), rgb_to_luma(target))

        pred_ds = downsample_for_loss(pred, self.loss_max_side)
        target_ds = downsample_for_loss(target, self.loss_max_side)

        if self.w_msssim > 0:
            levels = min(self.msssim_levels, 5)
            mss = 1.0 - msssim(pred_ds, target_ds, levels=levels)
        else:
            mss = pred.new_tensor(0.0)

        if self.w_fft > 0:
            fft = F.l1_loss(fft_mag(pred_ds), fft_mag(target_ds))
        else:
            fft = pred.new_tensor(0.0)

        if self.perceptual is not None:
            if next(self.perceptual.parameters()).device.type == "cpu":
                pf, tf = pred_ds.float().cpu(), target_ds.float().cpu()
                perc = self.perceptual(pf, tf).to(pred.device)
            else:
                perc = self.perceptual(pred_ds.float(), target_ds.float())
        else:
            perc = pred.new_tensor(0.0)

        total = (
            self.w_char * char
            + self.w_l1 * l1
            + self.w_luma * luma
            + self.w_msssim * mss
            + self.w_perc * perc
            + self.w_fft * fft
        )
        log = {
            "loss": float(total.detach()),
            "char": float(char.detach()),
            "l1": float(l1.detach()),
            "luma": float(luma.detach()),
            "msssim": float(mss.detach()),
            "fft": float(fft.detach()),
            "perc": float(perc.detach()) if torch.is_tensor(perc) else float(perc),
        }
        return total, log

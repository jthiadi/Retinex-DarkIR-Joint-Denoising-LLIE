"""
Paper (DarkIR, CVPR 2025):
  L = λp·L1 + λpe·LPIPS + λed·L_edge + L_lol
  λp=1, λpe=1e-2, λed=50

Our extension (zero inference cost):
  Heteroscedastic weights w = 1 + (1 - luma)² on L1 and L_lol for dark pixels.
  Optional: (1 - MS-SSIM) and FFT magnitude L1 for structure / frequency fidelity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def hetero_weights(low: torch.Tensor) -> torch.Tensor:
    lum = rgb_to_luma(low)
    return 1.0 + (1.0 - lum).pow(2)


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gradient L2 (Eq. 7 in DarkIR paper)."""
    px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(px, tx) + F.mse_loss(py, ty)


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
        s = s.clamp(min=1e-6, max=1.0)
        val = val * s ** weights[i]
    return val.mean()


def fft_mag(x: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_luma(x)
    f = torch.fft.rfft2(gray, norm="ortho")
    return torch.log1p(f.abs())


def downsample_for_loss(x: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return x
    h, w = x.shape[-2:]
    m = max(h, w)
    if m <= max_side:
        return x
    scale = max_side / m
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)


class DarkIRLLIELoss(nn.Module):
    """Composite loss for DarkIR-LLIE finetuning."""

    def __init__(
        self,
        w_l1: float = 1.0,
        w_lol: float = 1.0,
        w_edge: float = 50.0,
        w_lpips: float = 1e-2,
        w_msssim: float = 0.0,
        w_fft: float = 0.0,
        hetero: bool = True,
        loss_max_side: int = 256,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_lol = w_lol
        self.w_edge = w_edge
        self.w_lpips = w_lpips
        self.w_msssim = w_msssim
        self.w_fft = w_fft
        self.hetero = hetero
        self.loss_max_side = loss_max_side

        self._lpips = None
        if w_lpips > 0:
            import lpips

            self._lpips = lpips.LPIPS(net="vgg", verbose=False)
            if device is not None:
                self._lpips = self._lpips.to(device)
            self._lpips.eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)

    def _pixel_loss(self, pred: torch.Tensor, target: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        w = hetero_weights(low) if self.hetero else 1.0
        return (w * charbonnier(pred - target)).mean()

    def _lol(
        self,
        side_pred: torch.Tensor,
        target: torch.Tensor,
        low: torch.Tensor,
    ) -> torch.Tensor:
        gt_down = F.interpolate(
            target, size=side_pred.shape[-2:], mode="bilinear", align_corners=False,
        )
        if self.hetero:
            w = F.interpolate(
                hetero_weights(low),
                size=side_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return (w * charbonnier(side_pred - gt_down)).mean()
        return charbonnier(side_pred - gt_down).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        low: torch.Tensor,
        side_pred: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = self._pixel_loss(pred, target, low)
        lol = self._lol(side_pred, target, low) if side_pred is not None else pred.new_tensor(0.0)
        edge = edge_loss(pred, target)

        if self._lpips is not None and self.w_lpips > 0:
            lp = self._lpips(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()
        else:
            lp = pred.new_tensor(0.0)

        if self.w_msssim > 0:
            p_ss = downsample_for_loss(pred, self.loss_max_side)
            t_ss = downsample_for_loss(target, self.loss_max_side)
            ms = 1.0 - msssim(p_ss, t_ss)
        else:
            ms = pred.new_tensor(0.0)

        if self.w_fft > 0:
            p_f = downsample_for_loss(pred, self.loss_max_side)
            t_f = downsample_for_loss(target, self.loss_max_side)
            fft = F.l1_loss(fft_mag(p_f), fft_mag(t_f))
        else:
            fft = pred.new_tensor(0.0)

        total = (
            self.w_l1 * l1
            + self.w_lol * lol
            + self.w_edge * edge
            + self.w_lpips * lp
            + self.w_msssim * ms
            + self.w_fft * fft
        )
        log = {
            "loss": float(total.detach()),
            "l1": float(l1.detach()),
            "lol": float(lol.detach()) if torch.is_tensor(lol) else 0.0,
            "edge": float(edge.detach()),
            "lpips": float(lp.detach()),
            "msssim": float(ms.detach()),
            "fft": float(fft.detach()),
        }
        return total, log

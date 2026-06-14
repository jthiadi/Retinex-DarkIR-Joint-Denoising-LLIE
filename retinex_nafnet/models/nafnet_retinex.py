"""Retinex-guided NAFNet"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = c * dw_expand
        ffn = c * ffn_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(self.sg(self.conv2(self.conv1(self.norm1(x)))))
        x = x + y * self.beta
        y = self.conv5(F.gelu(self.conv4(self.norm2(x))))
        return x + y * self.gamma


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


from torch.utils.checkpoint import checkpoint


def _make_blocks(c: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(c) for _ in range(n)])


def _run_blocks(blocks: nn.Sequential, h: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
    if use_checkpoint and h.requires_grad:
        for block in blocks:
            h = checkpoint(block, h, use_reentrant=False)
        return h
    for block in blocks:
        h = block(h)
    return h


def _run_seq(blocks: nn.Sequential, h: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
    return _run_blocks(blocks, h, use_checkpoint)


class RetinexNAFNet(nn.Module):
    def __init__(
        self,
        width: int = 48,
        enc_blocks: tuple[int, ...] = (2, 2, 4, 6),
        dec_blocks: tuple[int, ...] = (2, 2, 2),
        middle_blocks: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.illum = IlluminationBranch()
        chs = [width, width * 2, width * 4, width * 8]

        self.intro = nn.Conv2d(6, chs[0], 3, padding=1)
        self.enc = nn.ModuleList([
            _make_blocks(chs[0], enc_blocks[0]),
            _make_blocks(chs[1], enc_blocks[1]),
            _make_blocks(chs[2], enc_blocks[2]),
            _make_blocks(chs[3], enc_blocks[3]),
        ])
        self.down = nn.ModuleList([
            nn.Conv2d(chs[0], chs[1], 2, stride=2),
            nn.Conv2d(chs[1], chs[2], 2, stride=2),
            nn.Conv2d(chs[2], chs[3], 2, stride=2),
        ])
        self.mid = _make_blocks(chs[3], middle_blocks)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2),
            nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2),
            nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2),
        ])
        self.dec = nn.ModuleList([
            _make_blocks(chs[2], dec_blocks[0]),
            _make_blocks(chs[1], dec_blocks[1]),
            _make_blocks(chs[0], dec_blocks[2]),
        ])
        self.out = nn.Conv2d(chs[0], 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illum = self.illum(x)
        # Small epsilon avoids div-by-zero / fp16 spikes in very dark regions.
        ref = (x / illum.clamp(min=1e-2)).clamp(0.0, 1.0)
        h = self.intro(torch.cat([ref, x], dim=1))

        skips = []
        for enc, down in zip(self.enc[:3], self.down):
            h = _run_seq(enc, h, self.use_checkpoint)
            skips.append(h)
            h = down(h)

        h = _run_seq(self.enc[3], h, self.use_checkpoint)
        h = _run_seq(self.mid, h, self.use_checkpoint)

        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = _run_seq(dec, h + skip, self.use_checkpoint)

        return (x + self.out(h)).clamp(0.0, 1.0)

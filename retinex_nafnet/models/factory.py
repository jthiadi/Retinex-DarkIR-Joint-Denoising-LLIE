"""Build restoration model from config"""

from __future__ import annotations

import torch.nn as nn

from .nafnet_retinex import RetinexNAFNet


def build_model(cfg: dict) -> nn.Module:
    mcfg = cfg["model"]

    return RetinexNAFNet(
        width=mcfg.get("width", 40),
        enc_blocks=tuple(mcfg.get("enc_blocks", (2, 2, 4, 4))),
        dec_blocks=tuple(mcfg.get("dec_blocks", (2, 2, 2))),
        middle_blocks=mcfg.get("middle_blocks", 6),
        use_checkpoint=bool(mcfg.get("use_checkpoint", False)),
    )

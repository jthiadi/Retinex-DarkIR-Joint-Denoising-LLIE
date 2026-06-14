from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """EMA weights on CPU to save VRAM; moved to GPU only for validation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().cpu()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data.detach().to(s.device), alpha=1.0 - d)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def to_eval_device(self, device: torch.device) -> torch.nn.Module:
        """Run validation/inference on GPU without keeping EMA resident there."""
        if device.type == "cpu":
            return self.shadow
        return self.shadow.to(device)

    def release_eval_device(self, device: torch.device) -> None:
        if device.type == "cuda":
            self.shadow.cpu()
            torch.cuda.empty_cache()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    step: int,
    stage: str,
    best_psnr: float,
    micro_step: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "stage": stage,
            "best_psnr": best_psnr,
            "micro_step": micro_step,
        },
        path,
    )


def load_state_dict_partial(
    model: torch.nn.Module,
    state: dict,
    label: str = "model",
) -> tuple[list[str], list[str]]:
    """Load matching keys only; skip shape/name mismatches (e.g. stage-1 → new stage-2)."""
    current = model.state_dict()
    loaded: dict = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        loaded[key] = value
    missing = [k for k in current if k not in loaded]
    model.load_state_dict(loaded, strict=False)
    if skipped:
        print(f"  {label}: skipped {len(skipped)} keys (architecture mismatch or obsolete)")
    if missing and len(loaded) < len(current):
        print(f"  {label}: {len(loaded)}/{len(current)} keys loaded ({len(missing)} fresh init)")
    return skipped, missing


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: EMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if strict:
        model.load_state_dict(ckpt["model"], strict=True)
        if ema is not None and ckpt.get("ema"):
            ema.shadow.load_state_dict(ckpt["ema"], strict=True)
    else:
        load_state_dict_partial(model, ckpt["model"], label="model")
        if ema is not None and ckpt.get("ema"):
            load_state_dict_partial(ema.shadow, ckpt["ema"], label="ema")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

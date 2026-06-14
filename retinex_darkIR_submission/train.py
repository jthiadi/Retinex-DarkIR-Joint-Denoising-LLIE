"""
  python train.py --config ablation_configs/A5_w16.yaml
  python train.py --config ablation_configs/A4_retinex_w32.yaml --no-retinex
  python finetune.py --config ablation_configs/A1_sideloss_w32.yaml
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
for p in (ROOT, ROOT / "archs", PROJ / "dataset"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from archs.DarkIR import DarkIR  # noqa: E402
from checkpoint_utils import load_darkir_checkpoint  # noqa: E402
from dataset import (  # noqa: E402
    PairedCompose,
    PairedLowLightDataset,
    PairedRandomCrop,
    PairedRandomFlip,
    PairedToTensor,
)
from losses import DarkIRLLIELoss  # noqa: E402


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def _resolve_path(path: str, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for candidate in (p, base / p, ROOT / p, PROJ / p):
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / p).resolve()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_psnr: float,
    best_lpips: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "best_lpips": best_lpips,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cuda",
    weights_only: bool = False,
) -> tuple[int, float, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    info = load_darkir_checkpoint(model, ckpt, device=device, strict=False)
    if info["missing"]:
        weights_only = True
        print(f"Loaded weights only (new keys: {len(info['missing'])})")
    if weights_only:
        return 0, 0.0, math.inf
    start_epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    best_psnr = float(ckpt.get("best_psnr", 0.0)) if isinstance(ckpt, dict) else 0.0
    best_lpips = float(ckpt.get("best_lpips", math.inf)) if isinstance(ckpt, dict) else math.inf
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            start_epoch = 0
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except ValueError:
            pass
    return start_epoch, best_psnr, best_lpips


def _forward(model: nn.Module, inp: torch.Tensor, side_loss: bool):
    if side_loss:
        side, pred = model(inp, side_loss=True)
        return side, torch.clamp(pred, 0.0, 1.0)
    pred = model(inp, side_loss=False)
    return None, torch.clamp(pred, 0.0, 1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    side_loss: bool,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float]]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for i, (inp, gt) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches and i >= max_batches:
            break
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        side, pred = _forward(model, inp, side_loss)
        if isinstance(criterion, DarkIRLLIELoss):
            loss, log = criterion(pred, gt, inp, side)
        else:
            loss = criterion(pred, gt)
            log = {"loss": float(loss.detach())}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for k, v in log.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return totals.get("loss", 0.0) / max(n, 1), {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lpips_fn: nn.Module | None,
    max_batches: int | None,
) -> dict[str, float]:
    from metrics import psnr as calc_psnr

    model.eval()
    psnrs: list[float] = []
    lpips_vals: list[float] = []
    for i, (inp, gt) in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches and i >= max_batches:
            break
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        _, pred = _forward(model, inp, side_loss=False)
        gt = gt.float()
        p = calc_psnr(pred, gt)
        if math.isfinite(p):
            psnrs.append(p)
        if lpips_fn is not None:
            l = lpips_fn(pred, gt)
            if math.isfinite(l):
                lpips_vals.append(l)
        if device.type == "cuda" and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
        gc.collect()
    n = len(psnrs)
    return {
        "psnr": sum(psnrs) / n if n else 0.0,
        "lpips": sum(lpips_vals) / len(lpips_vals) if lpips_vals else math.inf,
        "n": n,
    }


def build_model(net_cfg: dict, use_retinex: bool | None) -> DarkIR:
    retinex = use_retinex if use_retinex is not None else bool(net_cfg.get("use_retinex", False))
    return DarkIR(
        img_channel=net_cfg["img_channels"],
        width=net_cfg["width"],
        middle_blk_num_enc=net_cfg["middle_blk_num_enc"],
        middle_blk_num_dec=net_cfg["middle_blk_num_dec"],
        enc_blk_nums=net_cfg["enc_blk_nums"],
        dec_blk_nums=net_cfg["dec_blk_nums"],
        dilations=net_cfg["dilations"],
        extra_depth_wise=net_cfg["extra_depth_wise"],
        use_retinex=retinex,
        illum_base=int(net_cfg.get("illum_base", 32)),
    )


def run_training(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_cfg = cfg["train"]
    net_cfg = cfg["network"]
    data_cfg = cfg["dataset"]
    save_cfg = cfg["save"]
    loss_cfg = cfg.get("loss", {})
    base = ROOT

    use_retinex = None if not args.no_retinex else False

    train_root = _resolve_path(args.train_dir or data_cfg["train_dir"], base)
    val_root = _resolve_path(args.val_dir or data_cfg["val_dir"], base)
    save_dir = Path(save_cfg["checkpoint_dir"])
    if not save_dir.is_absolute():
        save_dir = (ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    side_loss = bool(loss_cfg)
    save_by = train_cfg.get("save_by", "psnr")

    val_max = args.val_max_batches
    if val_max is None:
        val_max = train_cfg.get("val_max_batches")
    if val_max == 0:
        val_max = None

    train_transform = PairedCompose([
        PairedRandomCrop(train_cfg["crop_size"]),
        PairedRandomFlip(p_h=0.5, p_v=0.0),
        PairedToTensor(),
    ])
    train_set = PairedLowLightDataset(train_root, transform=train_transform)
    val_set = PairedLowLightDataset(val_root, transform=None)
    num_workers = train_cfg.get("num_workers", 0 if sys.platform == "win32" else 2)

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Train: {train_root} | Val: {val_root}")
    print(f"Config: {cfg_path.name} | Retinex: {use_retinex if use_retinex is not None else net_cfg.get('use_retinex')}")

    model = build_model(net_cfg, use_retinex).to(device)

    if side_loss:
        criterion: nn.Module = DarkIRLLIELoss(
            w_l1=float(loss_cfg.get("w_l1", 1.0)),
            w_lol=float(loss_cfg.get("w_lol", 1.0)),
            w_edge=float(loss_cfg.get("w_edge", 50.0)),
            w_lpips=float(loss_cfg.get("w_lpips", 1e-2)),
            hetero=bool(loss_cfg.get("hetero", True)),
            device=device,
        )
    else:
        criterion = CharbonnierLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr_initial"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["eta_min"],
    )

    start_epoch = 0
    best_psnr = 0.0
    best_lpips = math.inf
    resume_path = args.resume or train_cfg.get("resume")
    if resume_path:
        rp = _resolve_path(str(resume_path), ROOT)
        w_only = bool(train_cfg.get("resume_weights_only", False))
        start_epoch, best_psnr, best_lpips = load_checkpoint(
            rp, model, optimizer, scheduler, device=str(device), weights_only=w_only,
        )
        print(f"Resumed: {rp}")

    lpips_val = None
    if side_loss and train_cfg.get("val_lpips"):
        from metrics import LPIPSMetric
        lpips_val = LPIPSMetric(net="vgg", device=device)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")
        train_loss, train_log = train_one_epoch(
            model, train_loader, criterion, optimizer, device, side_loss,
            train_cfg.get("train_max_batches") or None,
        )
        metrics = validate(model, val_loader, device, lpips_val, val_max)
        scheduler.step()
        print(
            f"train={train_loss:.4f} val_psnr={metrics['psnr']:.2f} "
            f"(N={metrics['n']}) best={save_by}"
        )
        save_checkpoint(
            save_dir / "latest_darkir.pth", model, optimizer, scheduler,
            epoch + 1, best_psnr, best_lpips,
        )
        improved = False
        if save_by == "lpips" and metrics["lpips"] < best_lpips:
            best_lpips = metrics["lpips"]
            improved = True
        elif save_by != "lpips" and metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            improved = True
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
        if improved:
            best_path = save_dir / "best_darkir.pth"
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch + 1, best_psnr, best_lpips,
            )
            print(f"Saved: {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Retinex-DarkIR ablations")
    parser.add_argument(
        "--config",
        default="ablation_configs/A0_baseline_w32.yaml",
        help="Path under ablation_configs/",
    )
    parser.add_argument("--no-retinex", action="store_true", help="Force vanilla 3-ch DarkIR")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()

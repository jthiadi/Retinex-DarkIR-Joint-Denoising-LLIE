"""Course grading score calculator for RetinexNAFNet submission.

Computes PSNR / SSIM / LPIPS on the val split and prints scores only.
Does not write enhanced images — use infer.py for that.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
for p in (ROOT, PROJ / "dataset"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
os.chdir(ROOT)

from dataset import PairedLowLightDataset  # noqa: E402
from metrics import LPIPSMetric, psnr  # noqa: E402
from models.factory import build_model  # noqa: E402
from train import _autocast, _split_dir  # noqa: E402
from utils import load_config  # noqa: E402
from losses import ssim_map  # noqa: E402

WEIGHTS = {"psnr": 0.3, "ssim": 0.3, "lpips": 0.4, "size": 0.5, "flops": 0.5}
DEFAULT_BOUNDS = {
    "psnr": {"min": 15.0, "max": 25.0, "higher_better": True},
    "ssim": {"min": 0.50, "max": 0.75, "higher_better": True},
    "lpips": {"min": 0.15, "max": 0.55, "higher_better": False},
    "params_m": {"min": 5.0, "max": 150.0, "higher_better": False},
    "gflops": {"min": 10.0, "max": 80.0, "higher_better": False},
}


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(ssim_map(pred, target).mean().item())


def rank_from_bounds(value: float, spec: dict) -> float:
    lo, hi = float(spec["min"]), float(spec["max"])
    t = max(0.0, min(1.0, (value - lo) / max(hi - lo, 1e-8)))
    return t if spec.get("higher_better", True) else 1.0 - t


def weighted_score(ranks: dict[str, float]) -> float:
    return sum(ranks[k] * WEIGHTS[k] for k in WEIGHTS)


def model_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def model_gflops(model: torch.nn.Module, input_size: int = 256) -> float:
    from thop import profile
    m = model.cpu().eval()
    x = torch.randn(1, 3, input_size, input_size)
    macs, _ = profile(m, inputs=(x,), verbose=False)
    return macs / 1e9


@torch.no_grad()
def run_val(config: Path, checkpoint: Path, device: torch.device, max_images: int | None) -> dict:
    cfg = load_config(str(config))
    val_root = _split_dir(cfg, "val_dir", "val")

    val_ds = PairedLowLightDataset(val_root, transform=None)
    if max_images:
        val_ds.stems = val_ds.stems[:max_images]
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    lpips_fn = LPIPSMetric(net="vgg", device=device)
    use_amp = device.type == "cuda" and cfg["train"].get("amp", True)
    amp_dtype = cfg["train"].get("amp_dtype", "bfloat16")

    psnrs, ssims, lpips_vals = [], [], []
    for low, high in tqdm(loader, desc="calc", unit="img"):
        low = low.to(device)
        high = high.to(device)
        with _autocast(use_amp, amp_dtype):
            pred = torch.clamp(model(low), 0.0, 1.0)
        pred = pred.float()
        high = high.float()
        psnrs.append(psnr(pred, high))
        ssims.append(_ssim(pred, high))
        lpips_vals.append(lpips_fn(pred, high))

    n = len(psnrs)
    return {
        "psnr": sum(psnrs) / n if n else float("nan"),
        "ssim": sum(ssims) / n if n else float("nan"),
        "lpips": sum(lpips_vals) / n if n else float("nan"),
        "n": n,
        "params_m": model_params_m(model),
        "gflops": model_gflops(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinexNAFNet grading calculator")
    parser.add_argument("--config", default="configs/rtx4050.yaml")
    parser.add_argument("--checkpoint", default="runs/21.0475.pt")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    ckpt = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    val_root = _split_dir(load_config(str(cfg)), "val_dir", "val")
    n_eval = args.max_images if args.max_images else len(PairedLowLightDataset(val_root))
    print(f"Computing val metrics on {n_eval} image(s). No files will be written (use infer.py for outputs).")

    raw = run_val(cfg, ckpt, device, args.max_images)
    ranks = {
        "psnr": rank_from_bounds(raw["psnr"], DEFAULT_BOUNDS["psnr"]),
        "ssim": rank_from_bounds(raw["ssim"], DEFAULT_BOUNDS["ssim"]),
        "lpips": rank_from_bounds(raw["lpips"], DEFAULT_BOUNDS["lpips"]),
        "size": rank_from_bounds(raw["params_m"], DEFAULT_BOUNDS["params_m"]),
        "flops": rank_from_bounds(raw["gflops"], DEFAULT_BOUNDS["gflops"]),
    }
    total = weighted_score(ranks)

    print("\n=== Raw metrics ===")
    print(f"  PSNR:  {raw['psnr']:.4f} dB  (N={raw['n']})")
    print(f"  SSIM:  {raw['ssim']:.4f}")
    print(f"  LPIPS: {raw['lpips']:.4f}")
    print(f"  Size:  {raw['params_m']:.2f} M params")
    print(f"  FLOPs: {raw['gflops']:.2f} GFlops @256x256x3")
    print(f"\n=== Weighted total: {total:.4f} / {sum(WEIGHTS.values()):.1f} ===")

    if args.save_json:
        args.save_json.write_text(
            json.dumps({"raw": raw, "ranks": ranks, "weighted_total": total}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

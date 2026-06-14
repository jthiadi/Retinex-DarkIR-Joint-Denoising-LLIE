from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
for p in (ROOT, ROOT / "archs", PROJ / "dataset"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from checkpoint_utils import load_darkir_checkpoint  # noqa: E402
from archs.DarkIR import DarkIR  # noqa: E402
from dataset import PairedLowLightDataset  # noqa: E402
from losses import ssim_map  # noqa: E402
from metrics import LPIPSMetric, psnr  # noqa: E402

WEIGHTS = {"psnr": 0.3, "ssim": 0.3, "lpips": 0.4, "size": 0.5, "flops": 0.5}
DEFAULT_BOUNDS = {
    "psnr": {"min": 15.0, "max": 25.0, "higher_better": True},
    "ssim": {"min": 0.50, "max": 0.75, "higher_better": True},
    "lpips": {"min": 0.15, "max": 0.55, "higher_better": False},
    "params_m": {"min": 0.5, "max": 40.0, "higher_better": False},
    "gflops": {"min": 1.0, "max": 80.0, "higher_better": False},
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


@torch.no_grad()
def run_val(
    checkpoint: Path,
    val_dir: Path,
    config: Path,
    device: torch.device,
    max_images: int | None,
    use_retinex: bool | None,
) -> dict[str, float]:
    with open(config, encoding="utf-8") as f:
        net_cfg = yaml.safe_load(f)["network"]
    model = build_model(net_cfg, use_retinex).to(device)
    load_darkir_checkpoint(model, checkpoint, device=str(device), strict=False)
    model.eval()

    ds = PairedLowLightDataset(val_dir, transform=None)
    if max_images:
        ds.stems = ds.stems[:max_images]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    lpips_fn = LPIPSMetric(net="vgg", device=device)
    psnrs, ssims, lpips_vals = [], [], []

    for i, (low, high) in enumerate(tqdm(loader, desc="calc", unit="img")):
        low = low.to(device)
        high = high.to(device)
        pred = torch.clamp(model(low, side_loss=False), 0.0, 1.0).float()
        high = high.float()
        psnrs.append(psnr(pred, high))
        ssims.append(_ssim(pred, high))
        lpips_vals.append(lpips_fn(pred, high))
        if device.type == "cuda" and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
        gc.collect()

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
    parser = argparse.ArgumentParser(description="DarkIR grading calculator")
    parser.add_argument("--config", default="ablation_configs/A2_edgeloss_w32.yaml")
    parser.add_argument("--checkpoint", default="result_checkpoints/retinex_DarkIR_L.pth")
    parser.add_argument("--val-dir", default="../dataset/val")
    parser.add_argument("--no-retinex", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    ckpt = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    val_dir = ROOT / args.val_dir if not Path(args.val_dir).is_absolute() else Path(args.val_dir)
    n_eval = args.max_images if args.max_images else len(PairedLowLightDataset(val_dir.resolve()))
    print(f"Computing val metrics on {n_eval} image(s). No files will be written (use infer.py for outputs).")

    raw = run_val(
        ckpt, val_dir.resolve(), cfg.resolve(), device, args.max_images,
        use_retinex=False if args.no_retinex else None,
    )
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

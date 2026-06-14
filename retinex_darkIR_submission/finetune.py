"""Finetune w=32 ablations A1–A4 from the vanilla baseline checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from train import run_training

ROOT = Path(__file__).resolve().parent
FINETUNE_CONFIGS = {
    "A1": "ablation_configs/A1_sideloss_w32.yaml",
    "A2": "ablation_configs/A2_edgeloss_w32.yaml",
    "A3": "ablation_configs/A3_LPIPS_w32.yaml",
    "A4": "ablation_configs/A4_retinex_w32.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune DarkIR ablations A1–A4")
    parser.add_argument("--config", default="A1", help="A1, A2, A3, A4 or yaml path")
    parser.add_argument("--no-retinex", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = FINETUNE_CONFIGS.get(args.config.upper(), args.config)
    if not str(cfg).endswith(".yaml"):
        raise SystemExit(f"Unknown config {args.config!r}. Use A1–A4 or a yaml path.")

    args.config = str(ROOT / cfg) if not Path(cfg).is_absolute() else cfg
    run_training(args)


if __name__ == "__main__":
    main()

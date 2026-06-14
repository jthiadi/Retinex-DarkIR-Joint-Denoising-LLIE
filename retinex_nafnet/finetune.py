from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune RetinexNAFNet for PSNR")
    parser.add_argument("--config", default="configs/finetune_psnr.yaml")
    parser.add_argument("--resume", default="runs/21.0475.pt")
    parser.add_argument("--best-name", default="best_ema_finetune.pt")
    parser.add_argument("--seed", type=int, default=42)
    args, extra = parser.parse_known_args()

    resume = Path(args.resume)
    if not resume.is_absolute():
        resume = ROOT / resume
    if not resume.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

    import train as train_mod

    orig_save = train_mod.save_checkpoint

    def _save(path, *a, **kw):
        if path.name == "best_ema.pt":
            path = path.parent / args.best_name
        return orig_save(path, *a, **kw)

    train_mod.save_checkpoint = _save

    argv = [
        str(ROOT / "train.py"),
        "--config",
        str(ROOT / args.config if not Path(args.config).is_absolute() else args.config),
        "--resume",
        str(resume),
        "--start-micro-step",
        "0",
        "--seed",
        str(args.seed),
        *extra,
    ]
    sys.argv = argv
    train_mod.main()


if __name__ == "__main__":
    main()

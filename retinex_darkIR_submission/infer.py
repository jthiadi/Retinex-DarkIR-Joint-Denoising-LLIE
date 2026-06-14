"""Run DarkIR inference on test images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
for p in (ROOT, ROOT / "archs", PROJ / "dataset"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from checkpoint_utils import load_darkir_checkpoint  # noqa: E402
from DarkIR import DarkIR  # noqa: E402
from dataset import TestLowLightDataset  # noqa: E402


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


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


def main() -> None:
    parser = argparse.ArgumentParser(description="DarkIR test inference")
    parser.add_argument("--config", default="ablation_configs/A2_edgeloss_w32.yaml")
    parser.add_argument("--checkpoint", default="result_checkpoints/retinex_DarkIR_L.pth")
    parser.add_argument("--input-dir", default="../dataset/test")
    parser.add_argument("--output-dir", default="outputs/retinex_DarkIR_L")
    parser.add_argument("--no-retinex", action="store_true")
    args = parser.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    ckpt_path = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    in_dir = ROOT / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    out_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, encoding="utf-8") as f:
        net_cfg = yaml.safe_load(f)["network"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_retinex = False if args.no_retinex else None
    model = build_model(net_cfg, use_retinex).to(device)
    load_darkir_checkpoint(model, ckpt_path, device=str(device), strict=False)
    model.eval()

    to_pil = transforms.ToPILImage()
    ds = TestLowLightDataset(in_dir, transform=None)

    with torch.no_grad():
        for x, stem in tqdm(ds, desc="infer"):
            x = x.unsqueeze(0).to(device)
            x, h, w = pad_to_multiple(x)
            pred = torch.clamp(model(x, side_loss=False), 0.0, 1.0)
            pred = pred[:, :, :h, :w]
            out_img = to_pil(pred.squeeze(0).cpu())
            out_img.save(out_dir / f"{stem}-out.webp", quality=95)

    print(f"Saved {len(ds)} images to {out_dir}")


if __name__ == "__main__":
    main()

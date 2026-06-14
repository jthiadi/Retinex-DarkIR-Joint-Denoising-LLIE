from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.factory import build_model
from utils import EMA, load_config


def load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=95)


def augment(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, [-1])
    if mode == 2:
        return torch.flip(x, [-2])
    if mode == 3:
        return torch.flip(x, [-1, -2])
    if mode == 4:
        return x.transpose(-1, -2)
    if mode == 5:
        return torch.flip(x.transpose(-1, -2), [-1])
    if mode == 6:
        return torch.flip(x.transpose(-1, -2), [-2])
    return torch.flip(x.transpose(-1, -2), [-1, -2])


def deaugment(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, [-1])
    if mode == 2:
        return torch.flip(x, [-2])
    if mode == 3:
        return torch.flip(x, [-1, -2])
    if mode == 4:
        return x.transpose(-1, -2)
    if mode == 5:
        return torch.flip(x, [-1]).transpose(-1, -2)
    if mode == 6:
        return torch.flip(x, [-2]).transpose(-1, -2)
    return torch.flip(x, [-1, -2]).transpose(-1, -2)


@torch.no_grad()
def predict_one(model: torch.nn.Module, x: torch.Tensor, tta: bool) -> torch.Tensor:
    if not tta:
        return model(x)
    acc = 0.0
    for m in range(8):
        xa = augment(x, m)
        ya = model(xa)
        acc = acc + deaugment(ya, m)
    return (acc / 8.0).clamp(0.0, 1.0)


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtx4050.yaml")
    parser.add_argument("--checkpoint", nargs="+", default=None)
    parser.add_argument("--input-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA not available — inference on CPU.")
    ckpts = [Path(p) for p in (args.checkpoint or [cfg["infer"]["checkpoint"]])]
    use_tta = args.tta or cfg["infer"].get("tta", False)

    models = [load_model(cfg, c, device) for c in ckpts if c.exists()]
    if not models:
        raise FileNotFoundError(f"No checkpoints found: {ckpts}")

    in_dir = root / args.input_dir
    out_dir = root / args.output_dir
    paths = sorted(in_dir.glob("*-in.webp"))
    print(f"Ensemble {len(models)} model(s) | TTA={use_tta} | {len(paths)} images")

    for path in tqdm(paths):
        x = load_image(path).to(device)
        acc = 0.0
        for model in models:
            acc = acc + predict_one(model, x, use_tta)
        out = (acc / len(models)).clamp(0.0, 1.0)
        out_name = path.name.replace("-in.webp", "-out.webp")
        save_image(out, out_dir / out_name)


if __name__ == "__main__":
    main()

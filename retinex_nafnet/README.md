# RetinexNAFNet 
Retinex-guided NAFNet with progressive multi-stage training (256 → 384 → 512 crops), EMA checkpoints, and optional TTA at inference.

Shared data in `../dataset/` (`train/`, `val/`, `test/`).

## Layout

```
retinex_nafnet/
  configs/
    baseline.yaml     full progressive training (s256 → s384 → s512)
    finetune.yaml     PSNR-focused s512 fine-tune
  models/             RetinexNAFNet architecture
  runs/               baseline.pt, finetuned.pt, …
  outputs/            enhanced test images from infer.py
  train.py
  finetune.py
  infer.py
  calculate.py        val metrics + grading score (no image output)
  requirements.txt
```

## Usage
```
pip install -r requirements.txt

python train.py --config configs/baseline.yaml
python finetune.py --config configs/finetune.yaml --resume runs/baseline.pt

python infer.py --config configs/baseline.yaml --checkpoint runs/baseline.pt --output-dir outputs
python infer.py --config configs/finetune.yaml --checkpoint runs/finetuned.pt --tta

python calculate.py --config configs/baseline.yaml --checkpoint runs/baseline.pt
python calculate.py --config configs/finetune.yaml --checkpoint runs/finetuned.pt
```

`calculate.py` prints PSNR / SSIM / LPIPS / size / FLOPs and a weighted score.

Optional flags:
- `train.py`: `--resume`, `--start-micro-step`, `--no-amp`, `--seed`
- `finetune.py`: `--best-name`, extra args forwarded to `train.py`
- `infer.py`: `--input-dir`, `--output-dir`, `--checkpoint` (repeat for ensemble)
- `calculate.py`: `--max-images`, `--save-json`

Grading formula:
```
PSNR × 0.3 + SSIM × 0.3 + LPIPS × 0.4 + Size × 0.5 + FLOPs × 0.5
```
(higher rank score is better; LPIPS / size / FLOPs are inverted in the script)

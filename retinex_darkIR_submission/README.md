# Retinex-DarkIR Submission

Low-light image enhancement with DarkIR (FreMLP + Di-SpAM), optional Retinex illumination branch, and ablation configs A0–A6.

Shared data lives in `../dataset/` (`train/`, `val/`, `test/`).

## Layout

```
retinex_darkIR_submission/
  ablation_configs/     A0–A6 yaml ablations
  archs/                DarkIR model code
  result_checkpoints/   retinex_DarkIR_M.pth, retinex_DarkIR_L.pth, …
  outputs/              enhanced test images from infer.py
  train.py              train from scratch or resume
  finetune.py           finetune A1–A4 from baseline
  infer.py              test inference → *-out.webp
  calculate.py          val metrics + grading score (no image output)
  requirements.txt
```

## Ablation configs

| Config | Loss stack | Width | Retinex |
|--------|-----------|-------|---------|
| A0 | hetero Charbonnier only | 32 | no |
| A1 | + side loss (L_lol) | 32 | no |
| A2 | + edge loss | 32 | no |
| A3 | + LPIPS | 32 | no |
| A4 | + Retinex | 32 | yes |
| A5 | side + edge (from scratch) | 16 | no |
| A6 | side + crop 320 | 16 | yes |

## Usage

From this folder, with the project venv activated:

```
pip install -r requirements.txt

python train.py --config ablation_configs/A5_w16.yaml
python train.py --config ablation_configs/A4_retinex_w32.yaml --no-retinex

python finetune.py --config A1
python finetune.py --config ablation_configs/A2_edgeloss_w32.yaml

python infer.py --config ablation_configs/A6_cropsize_w16.yaml --checkpoint result_checkpoints/retinex_DarkIR_M.pth --output-dir outputs/retinex_DarkIR_M
python infer.py --config ablation_configs/A2_edgeloss_w32.yaml --checkpoint result_checkpoints/retinex_DarkIR_L.pth --output-dir outputs/retinex_DarkIR_L

python calculate.py --config ablation_configs/A6_cropsize_w16.yaml --checkpoint result_checkpoints/retinex_DarkIR_M.pth
python calculate.py --config ablation_configs/A2_edgeloss_w32.yaml --checkpoint result_checkpoints/retinex_DarkIR_L.pth
```

`calculate.py` prints PSNR / SSIM / LPIPS / size / FLOPs and a weighted score. It does **not** write enhanced images — use `infer.py` for that.

Optional flags: `--no-retinex`, `--resume`, `--train-dir`, `--val-dir`, `--max-images`, `--save-json`.

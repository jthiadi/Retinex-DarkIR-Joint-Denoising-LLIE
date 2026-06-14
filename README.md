# LLIE Multimedia Project

## Layout

```
LLIE_Multimedia/
  dataset/
    train/          paired *-in.webp / *-gt.webp
    val/
    test/           *-in.webp only
    dataset.py
  retinex_darkIR_submission/
    ablation_configs/   A0–A6 yaml ablations
    archs/
    result_checkpoints/ retinex_DarkIR_M.pth, retinex_DarkIR_L.pth
    outputs/
    train.py, finetune.py, infer.py, calculate.py
  retinex_nafnet/
    configs/            baseline.yaml, finetune.yaml
    runs/               baseline.pt, finetuned.pt
    train.py, finetune.py, infer.py, calculate.py
  NAFNET/
    train.py, nafnet.py, losses.py
```

## Retinex-DarkIR
```
cd retinex_darkIR_submission
python train.py --config ablation_configs/A5_w16.yaml
python finetune.py --config A1
python infer.py --config ablation_configs/A6_cropsize_w16.yaml --checkpoint result_checkpoints/retinex_DarkIR_M.pth --output-dir outputs/retinex_DarkIR_M
python infer.py --config ablation_configs/A2_edgeloss_w32.yaml --checkpoint result_checkpoints/retinex_DarkIR_L.pth --output-dir outputs/retinex_DarkIR_L
python calculate.py --config ablation_configs/A6_cropsize_w16.yaml --checkpoint result_checkpoints/retinex_DarkIR_M.pth
python calculate.py --config ablation_configs/A2_edgeloss_w32.yaml --checkpoint result_checkpoints/retinex_DarkIR_L.pth
```

## RetinexNAFNet
```
cd retinex_nafnet
python train.py --config configs/baseline.yaml
python finetune.py --config configs/finetune.yaml --resume checkpoint/baseline.pt
python infer.py --checkpoint checkpoint/finetuned.pt
python calculate.py --checkpoint checkpoint/finetuned.pt
```

See each folder’s `README.md` for full details.
# NAFNet Baseline

Vanilla NAFNet training script (no Retinex). Uses the shared dataset loader in `../dataset/dataset.py`.

## Layout

```
NAFNET/
  nafnet.py       NAFNet architecture
  losses.py       L1 + MSE composite loss
  train.py        curriculum training (crop 128 → 192 → 256)
  calculate.py    val metrics + grading score (no image output)
  checkpoints/    best weights (aminbagus.pth)
```

Data paths (inside `train.py`):

```
../dataset/train
../dataset/val
```

## Usage

```
pip install torch torchvision pillow tqdm pytorch-msssim lpips thop

python train.py
python calculate.py --checkpoint checkpoints/aminbagus.pth
python calculate.py --max-images 20
```

Training saves the best val-PSNR checkpoint to `checkpoints/aminbagus.pth`.

Hyperparameters are set at the top of `train.py` (batch size 12, grad accum 2, 500 epochs, AdamW lr 2e-4). Crop size increases at epochs 1 / 100 / 300.
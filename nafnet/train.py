import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
for p in (PROJ / "dataset",):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from dataset import (
    PairedLowLightDataset,
    PairedCompose,
    PairedRandomCrop,
    PairedRandomFlip,
    PairedToTensor,
    PairedRandomRotate,
)

from nafnet import NAFNet
from losses import TotalLoss

import torch.nn.functional as F


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
        }

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(p.data, alpha=1 - self.decay)

    def copy_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])


def calc_psnr(pred, target):
    pred = torch.nan_to_num(pred)
    target = torch.nan_to_num(target)
    mse = F.mse_loss(pred, target)
    return -10 * torch.log10(mse + 1e-8)


def make_loader(dataset, patch_size, batch_size, train=True):
    """
    Rebuild loader with a new crop size (for curriculum).
    """
    if train:
        tf = PairedCompose([
            PairedRandomCrop(patch_size),
            PairedRandomFlip(0.5, 0.5),
            PairedRandomRotate(),
            PairedToTensor(),
        ])

        dataset.transform = tf

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=6,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )


def main():

    # ── Config ───────────────────────────────────────────────────────────────
    DEVICE = "cuda"
    BATCH_SIZE = 12
    GRAD_ACCUM = 2
    LR = 2e-4
    EPOCHS = 500

    SAVE_DIR = Path("checkpoints")
    SAVE_DIR.mkdir(exist_ok=True)

    torch.backends.cudnn.benchmark = True

    print(f"Using: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_set = PairedLowLightDataset(
        PROJ / "dataset" / "train",
        transform=None
    )

    val_set = PairedLowLightDataset(
        PROJ / "dataset" / "val",
        transform=None
    )

    train_loader = make_loader(
        train_set,
        patch_size=128,
        batch_size=BATCH_SIZE,
        train=True
    )

    val_loader = make_loader(
        val_set,
        patch_size=None,
        batch_size=1,
        train=False
    )

    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NAFNet(
        base_ch=64,
        enc_blocks=[2, 2, 4, 8],
        dec_blocks=[2, 2, 2, 2],
    ).to(DEVICE)

    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model, backend="eager")

    ema = EMA(model, decay=0.999)

    criterion = TotalLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=100,
        T_mult=2,
        eta_min=1e-7,
    )

    scaler = torch.amp.GradScaler('cuda')

    best_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ── Curriculum ──────────────────────────────────────────────────────
        if epoch == 1:
            train_loader = make_loader(
                train_set,
                128,
                BATCH_SIZE,
                train=True
            )

        elif epoch == 100:
            train_loader = make_loader(
                train_set,
                192,
                BATCH_SIZE,
                train=True
            )

        elif epoch == 300:
            train_loader = make_loader(
                train_set,
                256,
                BATCH_SIZE,
                train=True
            )

        model.train()

        total_loss = 0.0

        optimizer.zero_grad()

        for i, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        ):

            x = x.to(
                DEVICE,
                non_blocking=True
            ).to(memory_format=torch.channels_last)

            y = y.to(
                DEVICE,
                non_blocking=True
            ).to(memory_format=torch.channels_last)

            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss, _ = criterion(pred, y)
                loss = loss / GRAD_ACCUM

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                scaler.step(optimizer)
                scaler.update()

                ema.update(model)

                optimizer.zero_grad()

            scheduler.step(
                epoch - 1 + i / len(train_loader)
            )

            total_loss += loss.item() * GRAD_ACCUM

        # ── Validation ──────────────────────────────────────────────────────
        raw_state = {
            k: v.clone()
            for k, v in model.state_dict().items()
        }

        ema.copy_to(model)

        model.eval()

        psnr_scores = []

        with torch.no_grad():
            for x, y in val_loader:

                x = x.to(
                    DEVICE,
                    non_blocking=True
                ).to(memory_format=torch.channels_last)

                y = y.to(
                    DEVICE,
                    non_blocking=True
                ).to(memory_format=torch.channels_last)

                with torch.amp.autocast('cuda'):
                    pred = torch.clamp(model(x), 0, 1)

                psnr_scores.append(
                    calc_psnr(pred, y).item()
                )

        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_loss = total_loss / len(train_loader)

        print(
            f"\nEpoch {epoch} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val PSNR: {avg_psnr:.2f} dB"
        )

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

            torch.save(
                model.state_dict(),
                SAVE_DIR / "aminbagus.pth"
            )

            print(
                f" ✓ Saved best model "
                f"(PSNR: {best_psnr:.2f} dB)"
            )

        model.load_state_dict(raw_state)


if __name__ == '__main__':
    main()
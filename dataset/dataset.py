"""Reference torch dataset for the low-light restoration splits.

Splits are produced by prepare_dataset.py (then re-encoded by reencode_webp.py)
and laid out as:
    train/, val/   ->  <stem>-in.webp  paired with  <stem>-gt.webp
    test/          ->  <stem>-in.webp  (no ground truth)

Two dataset classes:
    PairedLowLightDataset(root, transform=None)
        For train/ and val/. Yields (input_tensor, gt_tensor) by default,
        or a dict if `transform` is a pair-aware callable (see below).

    TestLowLightDataset(root, transform=None)
        For test/. Yields (input_tensor, stem).

Custom transform contract
-------------------------
A transform may be one of:

  1. None
        Inputs/GTs are converted to float32 CHW tensors in [0, 1].

  2. A torchvision-style callable on a single PIL image
        Applied independently to the input and (if present) the GT.
        Use this only for deterministic ops (e.g. ToTensor, Normalize).
        Randomized single-image transforms will desync the pair.

  3. A pair-aware callable: fn(input_pil, gt_pil) -> (input, gt)
        Marked by setting `fn.paired = True`. The callable owns randomness
        and must apply the same geometric augmentation to both images.
        See `PairedRandomCrop` and `PairedCompose` below for examples.
"""

import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

PairTransform = Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]
SingleTransform = Callable[[Image.Image], torch.Tensor]
Transform = Union[PairTransform, SingleTransform, None]


def _is_paired(t: Transform) -> bool:
    return t is not None and getattr(t, "paired", False)


def _default_to_tensor(img: Image.Image) -> torch.Tensor:
    return TF.to_tensor(img)  # float32 CHW in [0, 1]


IMG_EXT = ".webp"


def _scan(root: Path, require_gt: bool) -> list[str]:
    suffix = f"-in{IMG_EXT}"
    stems = []
    for p in sorted(root.glob(f"*{suffix}")):
        stem = p.name[: -len(suffix)]
        if require_gt and not (root / f"{stem}-gt{IMG_EXT}").exists():
            continue
        stems.append(stem)
    if not stems:
        raise RuntimeError(f"no samples found under {root}")
    return stems


class PairedLowLightDataset(Dataset):
    """train/ and val/ splits — input + gt pairs."""

    def __init__(self, root: Union[str, Path], transform: Transform = None):
        self.root = Path(root)
        self.transform = transform
        self.stems = _scan(self.root, require_gt=True)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img_in = Image.open(self.root / f"{stem}-in{IMG_EXT}").convert("RGB")
        img_gt = Image.open(self.root / f"{stem}-gt{IMG_EXT}").convert("RGB")

        if _is_paired(self.transform):
            return self.transform(img_in, img_gt)
        if self.transform is not None:
            return self.transform(img_in), self.transform(img_gt)
        return _default_to_tensor(img_in), _default_to_tensor(img_gt)


class TestLowLightDataset(Dataset):
    """test/ split — input only. Yields (tensor, stem)."""

    def __init__(self, root: Union[str, Path], transform: SingleTransform | None = None):
        self.root = Path(root)
        if _is_paired(transform):
            raise ValueError("test set has no GT; pair-aware transforms are not allowed")
        self.transform = transform
        self.stems = _scan(self.root, require_gt=False)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        stem = self.stems[idx]
        img = Image.open(self.root / f"{stem}-in{IMG_EXT}").convert("RGB")
        tensor = self.transform(img) if self.transform is not None else _default_to_tensor(img)
        return tensor, stem


# ---------------------------------------------------------------------------
# Pair-aware transform building blocks. Mark callables with `paired = True`
# so the dataset routes both images through them together.
# ---------------------------------------------------------------------------

class PairedCompose:
    paired = True

    def __init__(self, ops):
        self.ops = ops

    def __call__(self, a, b):
        for op in self.ops:
            a, b = op(a, b) if getattr(op, "paired", False) else (op(a), op(b))
        return a, b


class PairedRandomCrop:
    paired = True

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img_in: Image.Image, img_gt: Image.Image):
        W, H = img_in.size
        if (W, H) != img_gt.size:
            raise ValueError("input/gt size mismatch")
        if W < self.size or H < self.size:
            raise ValueError(f"image {W}x{H} smaller than crop {self.size}")
        x = random.randint(0, W - self.size)
        y = random.randint(0, H - self.size)
        box = (x, y, x + self.size, y + self.size)
        return img_in.crop(box), img_gt.crop(box)


class PairedRandomFlip:
    paired = True

    def __init__(self, p_h: float = 0.5, p_v: float = 0.0):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, a, b):
        if random.random() < self.p_h:
            a = TF.hflip(a)
            b = TF.hflip(b)
        if random.random() < self.p_v:
            a = TF.vflip(a)
            b = TF.vflip(b)
        return a, b


class PairedToTensor:
    paired = True

    def __call__(self, a, b):
        return _default_to_tensor(a), _default_to_tensor(b)


class PairedCenterCrop:
    paired = True

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img_in: Image.Image, img_gt: Image.Image):
        W, H = img_in.size
        if (W, H) != img_gt.size:
            raise ValueError("input/gt size mismatch")
        if self.size >= min(W, H):
            return img_in, img_gt
        x = (W - self.size) // 2
        y = (H - self.size) // 2
        box = (x, y, x + self.size, y + self.size)
        return img_in.crop(box), img_gt.crop(box)


class PairedRandomRotate:
    paired = True

    def __call__(self, a: Image.Image, b: Image.Image):
        k = random.randint(0, 3)
        if k == 1:
            a = a.transpose(Image.ROTATE_90)
            b = b.transpose(Image.ROTATE_90)
        elif k == 2:
            a = a.transpose(Image.ROTATE_180)
            b = b.transpose(Image.ROTATE_180)
        elif k == 3:
            a = a.transpose(Image.ROTATE_270)
            b = b.transpose(Image.ROTATE_270)
        return a, b


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parent

    train_tf = PairedCompose([
        PairedRandomCrop(256),
        PairedRandomFlip(p_h=0.5),
        PairedToTensor(),
    ])

    train_set = PairedLowLightDataset(root / "train", transform=train_tf)
    val_set   = PairedLowLightDataset(root / "val",   transform=None)
    test_set  = TestLowLightDataset(root / "test",    transform=None)

    print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

    loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    x, y = next(iter(loader))
    print(f"batch x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")


from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
for p in (ROOT, PROJ / "dataset"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
os.chdir(ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (  
    PairedCenterCrop,
    PairedCompose,
    PairedLowLightDataset,
    PairedRandomCrop,
    PairedRandomFlip,
    PairedRandomRotate,
    PairedToTensor,
)
from losses import RestorationLoss
from metrics import psnr, ssim
from models.factory import build_model
from utils import EMA, load_config, save_checkpoint, set_seed

try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = None


def _data_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    return root


def _split_dir(cfg: dict, key: str, default: str) -> Path:
    return (_data_root(cfg) / cfg["data"].get(key, default)).resolve()


def setup_cuda() -> None:
    if not torch.cuda.is_available():
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def model_finite(m: torch.nn.Module) -> bool:
    for p in m.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"{alloc:.0f}MB used / {reserved:.0f}MB reserved"


def _autocast(enabled: bool, dtype_name: str | None = None):
    if not enabled:
        if _AMP_DEVICE is not None:
            return autocast(_AMP_DEVICE, enabled=False)
        return autocast(enabled=False)
    if _AMP_DEVICE is None:
        return autocast(enabled=False)
    dn = (dtype_name or "float16").lower()
    if dn == "bfloat16" and torch.cuda.is_bf16_supported():
        return autocast(_AMP_DEVICE, dtype=torch.bfloat16, enabled=True)
    return autocast(_AMP_DEVICE, dtype=torch.float16, enabled=True)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    amp_dtype: str | None = None,
) -> dict[str, float]:
    model.eval()
    psnrs, ssims = [], []
    use_amp = device.type == "cuda"
    for i, (low, high) in enumerate(loader):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        with _autocast(use_amp, amp_dtype):
            pred = model(low)
        p = psnr(pred.float(), high)
        s = ssim(pred.float(), high)
        if math.isfinite(p):
            psnrs.append(p)
        if math.isfinite(s):
            ssims.append(s)
        if max_batches > 0 and i + 1 >= max_batches:
            break
    if not psnrs:
        return {"psnr": float("nan"), "ssim": float("nan")}
    return {"psnr": sum(psnrs) / len(psnrs), "ssim": sum(ssims) / len(ssims)}


def _ema_for_eval(ema: EMA, device: torch.device) -> torch.nn.Module:
    """Move EMA to GPU for validation; works with old and new utils.EMA."""
    if hasattr(ema, "to_eval_device"):
        return ema.to_eval_device(device)
    if device.type == "cpu":
        return ema.shadow
    return ema.shadow.to(device)


def _ema_release(ema: EMA, device: torch.device) -> None:
    if hasattr(ema, "release_eval_device"):
        ema.release_eval_device(device)
        return
    if device.type == "cuda" and next(ema.shadow.parameters()).device.type == "cuda":
        ema.shadow.cpu()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtx4050.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--start-micro-step",
        type=int,
        default=None,
        help="Skip micro-batches at the start of the first training stage (e.g. 92000 for s256)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Full FP32 (slower, most stable)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    setup_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA not available — training on CPU (install PyTorch with CUDA).")

    train_root = _split_dir(cfg, "train_dir", "train")
    val_root = _split_dir(cfg, "val_dir", "val")
    if not train_root.exists():
        raise FileNotFoundError(f"Training split not found: {train_root}")
    if not val_root.exists():
        print(f"WARNING: val split not found at {val_root} — using train split for validation")
        val_root = train_root
    print(f"Training: {train_root}")
    print(f"Validation: {val_root}")

    mcfg = cfg["model"]
    model = build_model(cfg).to(device)
    mtype = mcfg.get("type", "retinex_nafnet")
    print(f"Model: {mtype} | width={mcfg.get('width')}")

    criterion = RestorationLoss(cfg["loss"]).to(device)
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.999))
    if criterion.perceptual is not None:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device.type == "cuda" else 0
        perc_on_cpu = cfg["loss"].get("perc_on_cpu")
        if perc_on_cpu is None:
            perc_on_cpu = device.type == "cuda" and vram_gb < 7.0
        if perc_on_cpu:
            criterion.perceptual.cpu()
            print("VGG perceptual on CPU (saves VRAM on 6 GB GPUs)")
        else:
            criterion.perceptual.to(device)

    global_step = 0
    best_psnr = 0.0
    resume_ckpt: dict | None = None
    run_dir = ROOT / "runs"
    run_dir.mkdir(exist_ok=True)

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        if resume_ckpt.get("ema"):
            ema.shadow.load_state_dict(resume_ckpt["ema"])
        global_step = resume_ckpt.get("step", 0)
        best_psnr = resume_ckpt.get("best_psnr", 0.0)
        print(
            f"Resumed weights from {args.resume} | global_step={global_step} "
            f"| best_psnr={best_psnr:.3f} | stage={resume_ckpt.get('stage', '?')}"
        )

    use_amp = cfg["train"].get("amp", True) and device.type == "cuda" and not args.no_amp
    amp_dtype = cfg["train"].get("amp_dtype", "bfloat16")
    if use_amp and amp_dtype and str(amp_dtype).lower() == "bfloat16" and torch.cuda.is_bf16_supported():
        print("AMP dtype: bfloat16 (recommended on RTX 40-series)")
    elif use_amp:
        print("AMP dtype: float16")
    scaler = GradScaler(_AMP_DEVICE or "cuda", enabled=use_amp) if _AMP_DEVICE else GradScaler(enabled=use_amp)

    val_crop = cfg["train"].get("val_crop", 384)
    val_max_batches = cfg["train"].get("val_max_batches", 40)
    train_eval_max_batches = cfg["train"].get("train_eval_max_batches")
    if train_eval_max_batches is None:
        train_eval_max_batches = val_max_batches if val_max_batches > 0 else 40
    empty_cache = cfg["train"].get("empty_cache_after_val", True)

    stages = cfg["train"]["stages"]
    for stage_idx, stage in enumerate(stages):
        crop = stage["crop"]
        batch_size = stage["batch_size"]
        grad_accum = stage.get("grad_accum", 1)
        steps = stage["steps"]
        lr = stage["lr"]
        name = stage["name"]

        start_micro = 0
        saved_stage = resume_ckpt.get("stage") if resume_ckpt else None
        saved_micro = int(resume_ckpt.get("micro_step") or 0) if resume_ckpt else 0
        stage_names = [s["name"] for s in stages]

        if resume_ckpt and saved_stage and saved_stage in stage_names:
            saved_idx = stage_names.index(saved_stage)
            cur_idx = stage_names.index(name)
            if cur_idx < saved_idx:
                print(f"Skipping stage {name} (checkpoint already at {saved_stage})")
                continue
            # Multi-stage only: same stage fully done in checkpoint. Single-stage configs
            if (
                len(stages) > 1
                and cur_idx == saved_idx
                and saved_micro >= steps
            ):
                print(f"Skipping stage {name} (already finished in checkpoint)")
                continue

        if resume_ckpt and saved_stage == name and 0 < saved_micro < steps:
            start_micro = saved_micro

        if stage_idx == 0 and args.start_micro_step is not None:
            start_micro = max(0, min(args.start_micro_step, steps - 1))
        elif stage_idx == 0 and resume_ckpt and resume_ckpt.get("micro_step") and saved_stage != name:
            start_micro = max(0, min(int(resume_ckpt["micro_step"]), steps - 1))

        if resume_ckpt and saved_stage == name and start_micro > 0:
            print(f"Resuming stage {name} from micro-step {start_micro}/{steps}")

        num_workers = int(cfg["train"].get("num_workers", 0))
        train_tf = PairedCompose([
            PairedRandomCrop(crop),
            PairedRandomFlip(p_h=0.5, p_v=0.5),
            PairedRandomRotate(),
            PairedToTensor(),
        ])
        eval_tf = PairedCompose([PairedCenterCrop(val_crop), PairedToTensor()])
        train_ds = PairedLowLightDataset(train_root, transform=train_tf)
        val_ds = PairedLowLightDataset(val_root, transform=eval_tf)
        train_eval_ds = PairedLowLightDataset(train_root, transform=eval_tf)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        last_train_psnr = float("nan")
        last_val_psnr = float("nan")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
        if resume_ckpt and resume_ckpt.get("optimizer") and stage_idx == 0 and start_micro > 0:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            except Exception:
                pass

        opt_steps = max(1, steps // grad_accum)
        start_opt = start_micro // grad_accum
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt_steps, last_epoch=max(-1, start_opt - 1)
        )

        eff_bs = batch_size * grad_accum
        print(
            f"\n=== Stage {name} | crop={crop} bs={batch_size}×{grad_accum}={eff_bs} "
            f"| micro {start_micro}/{steps} | val_crop={val_crop} | GPU {gpu_mem_mb()} ==="
        )
        pbar = tqdm(range(start_micro, steps), desc=name, initial=start_micro, total=steps)
        train_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        for step_i in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            low = batch[0].to(device, non_blocking=True)
            high = batch[1].to(device, non_blocking=True)

            model.train()
            with _autocast(use_amp, amp_dtype if use_amp else None):
                pred = model(low)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    loss, logs = criterion(pred.float(), high.float(), low.float())
            else:
                loss, logs = criterion(pred, high, low)
            loss = loss / grad_accum

            if not torch.isfinite(loss).all():
                pbar.write(
                    "ERROR: non-finite loss. Stop training, then try: "
                    "`--no-amp`, set loss.w_perc to 0.0, or lower learning rates in the YAML."
                )
                raise RuntimeError("Non-finite loss")

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = (step_i + 1) % grad_accum == 0
            if do_step:
                if use_amp:
                    if cfg["train"].get("grad_clip"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg["train"].get("grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1

                if global_step % cfg["train"].get("log_every", 200) == 0:
                    if device.type == "cuda":
                        gc.collect()
                    pbar.set_postfix(
                        loss=f"{logs['loss']:.4f}",
                        tr_psnr=f"{last_train_psnr:.2f}" if math.isfinite(last_train_psnr) else "—",
                        val_psnr=f"{last_val_psnr:.2f}" if math.isfinite(last_val_psnr) else "—",
                        best=f"{best_psnr:.2f}",
                        mem=gpu_mem_mb(),
                    )

                if global_step % cfg["train"].get("val_every", 2000) == 0:
                    amp_eval = amp_dtype if use_amp else None
                    eval_model = _ema_for_eval(ema, device)
                    train_metrics = validate(
                        eval_model, train_eval_loader, device, train_eval_max_batches, amp_eval
                    )
                    val_metrics = validate(
                        eval_model, val_loader, device, val_max_batches, amp_eval
                    )
                    _ema_release(ema, device)
                    last_train_psnr = train_metrics["psnr"]
                    last_val_psnr = val_metrics["psnr"]
                    train_note = (
                        f"train PSNR={train_metrics['psnr']:.3f} SSIM={train_metrics['ssim']:.4f}"
                    )
                    if train_eval_max_batches > 0:
                        train_note += f" (first {train_eval_max_batches} batches)"
                    val_note = f"val PSNR={val_metrics['psnr']:.3f} SSIM={val_metrics['ssim']:.4f}"
                    if val_max_batches > 0:
                        val_note += f" (first {val_max_batches} batches)"
                    pbar.write(f"[step {global_step}] {train_note} | {val_note} | {gpu_mem_mb()}")
                    if math.isfinite(val_metrics["psnr"]) and val_metrics["psnr"] > best_psnr:
                        best_psnr = val_metrics["psnr"]
                        if model_finite(eval_model):
                            save_checkpoint(
                                run_dir / "best_ema.pt",
                                model,
                                ema,
                                optimizer,
                                global_step,
                                name,
                                best_psnr,
                                micro_step=step_i + 1,
                            )
                        else:
                            pbar.write("WARNING: EMA has NaN/Inf — not saving best_ema.pt")
                    model.train()
                    if empty_cache and device.type == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()

                if global_step % cfg["train"].get("save_every", 5000) == 0:
                    save_checkpoint(
                        run_dir / "last.pt",
                        model,
                        ema,
                        optimizer,
                        global_step,
                        name,
                        best_psnr,
                        micro_step=step_i + 1,
                    )

        save_checkpoint(
            run_dir / f"stage_{name}.pt",
            model,
            ema,
            optimizer,
            global_step,
            name,
            best_psnr,
            micro_step=steps,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Do not overwrite best_ema.pt here — final weights may be worse or numerically unstable.
    save_checkpoint(run_dir / "final.pt", model, ema, optimizer, global_step, "final", best_psnr)
    print(f"Done. Best val PSNR: {best_psnr:.3f} | last weights: runs/final.pt | best checkpoint: runs/best_ema.pt")


if __name__ == "__main__":
    main()

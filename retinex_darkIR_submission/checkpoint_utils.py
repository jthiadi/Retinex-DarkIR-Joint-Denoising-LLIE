"""Checkpoint loading with intro-channel adaptation (3-ch -> 4-ch / 6-ch, width shrink)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def extract_state_dict(ckpt) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        return ckpt
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("params")
        or ckpt.get("state_dict")
        or ckpt.get("ema")
        or ckpt.get("model")
        or ckpt
    )
    return {k.removeprefix("module."): v for k, v in state.items()}


def adapt_intro_weights(state: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapt intro conv: 3->4/6 input channels and/or w=32->w=16 output channels."""
    w_key = "intro.weight"
    if w_key not in state:
        return state

    old_w = state[w_key]
    new_w = model.state_dict()[w_key]
    if old_w.shape == new_w.shape:
        return state

    state = dict(state)
    out_new, in_new = new_w.shape[0], new_w.shape[1]
    old_w = old_w[:out_new]

    if old_w.shape[1] == 3 and in_new == 4:
        adapted = new_w.clone()
        adapted[:, 0:3] = old_w
        adapted[:, 3:4] = old_w.mean(dim=1, keepdim=True)
        state[w_key] = adapted
    elif old_w.shape[1] == 3 and in_new == 6:
        adapted = new_w.clone()
        adapted[:, 3:6] = old_w
        adapted[:, 0:3] = old_w * 0.5
        state[w_key] = adapted
    elif old_w.shape[1] == in_new:
        state[w_key] = old_w

    b_key = "intro.bias"
    if b_key in state and b_key in model.state_dict():
        old_b = state[b_key]
        new_b = model.state_dict()[b_key]
        if old_b.shape != new_b.shape and old_b.shape[0] >= new_b.shape[0]:
            state[b_key] = old_b[: new_b.shape[0]]
    return state


def filter_compatible_state(
    state: dict[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    """Keep only keys whose tensor shapes match the target model (e.g. w=32 ckpt -> w=16)."""
    model_sd = model.state_dict()
    return {k: v for k, v in state.items() if k in model_sd and v.shape == model_sd[k].shape}


def adapt_retinex_intro(state: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    return adapt_intro_weights(state, model)


def load_darkir_checkpoint(
    model: nn.Module,
    ckpt_path_or_dict,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> dict:
    if isinstance(ckpt_path_or_dict, (str, Path)):
        ckpt = torch.load(ckpt_path_or_dict, map_location=device, weights_only=False)
    else:
        ckpt = ckpt_path_or_dict
    state = adapt_intro_weights(extract_state_dict(ckpt), model)
    state = filter_compatible_state(state, model)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": missing, "unexpected": unexpected, "loaded_keys": len(state)}

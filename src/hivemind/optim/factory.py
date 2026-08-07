"""Optimizer factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .masked_adamw import MaskedAdamW

# Config key suffixes for per-group overrides, e.g. lr_lora / weight_decay_base.
_GROUP_SUFFIX = {"P": "base", "F": "lora", "shared": "shared"}


def build_optimizer(
    model: nn.Module,
    cfg: dict[str, Any],
) -> MaskedAdamW:
    """Build a MaskedAdamW optimizer from config.

    Parameters are split into P (base), F (LoRA), and shared groups via the
    model's ``get_param_groups()`` so base and fast memory can use different
    learning rates / weight decay (``lr_base``, ``lr_lora``, ``lr_shared``,
    ``weight_decay_base``, ...), each falling back to the global ``lr`` /
    ``weight_decay``.

    Args:
        model: the student model (must expose ``get_param_groups()``).
        cfg: optimizer config dict with keys: lr, betas, weight_decay, eps,
            amsgrad, plus optional per-group overrides.

    Returns:
        MaskedAdamW optimizer instance with named param groups.
    """
    lr = cfg.get("lr", 3e-4)
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    weight_decay = cfg.get("weight_decay", 0.01)
    eps = cfg.get("eps", 1e-8)
    amsgrad = bool(cfg.get("amsgrad", False))

    # Every other consumer of the student fails loudly on a DDP wrapper
    # (no __getattr__ forwarding), but this one would not: hasattr() is
    # False, so it would quietly collapse to a single "shared" group and
    # discard lr_lora / lr_base / weight_decay_base — a silent change of
    # experiment with no traceback.
    if isinstance(model, nn.parallel.DistributedDataParallel):
        raise TypeError(
            "build_optimizer requires the unwrapped student module. A "
            "DistributedDataParallel wrapper has no get_param_groups(), so the "
            "P/F/shared learning-rate groups would be silently collapsed into "
            "one. Pass ddp.module (or the raw student you wrapped)."
        )

    if hasattr(model, "get_param_groups"):
        named = model.get_param_groups()
    else:
        named = {"shared": list(model.parameters())}

    param_groups = []
    for name, params in named.items():
        trainable = [p for p in params if p.requires_grad]
        if not trainable:
            continue
        suffix = _GROUP_SUFFIX.get(name, name)
        param_groups.append(
            {
                "params": trainable,
                "lr": cfg.get(f"lr_{suffix}", lr),
                "weight_decay": cfg.get(f"weight_decay_{suffix}", weight_decay),
                "name": name,
            }
        )
    if not param_groups:
        raise ValueError("Model has no trainable parameters")

    return MaskedAdamW(
        param_groups,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
    )

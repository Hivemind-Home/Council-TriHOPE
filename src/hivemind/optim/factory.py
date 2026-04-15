"""Optimizer factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .masked_adamw import MaskedAdamW


def build_optimizer(
    model: nn.Module,
    cfg: dict[str, Any],
) -> MaskedAdamW:
    """Build a MaskedAdamW optimizer from config.

    Separates parameter groups for different learning rates if needed.

    Args:
        model: the student model.
        cfg: optimizer config dict with keys: lr, betas, weight_decay.

    Returns:
        MaskedAdamW optimizer instance.
    """
    lr = cfg.get("lr", 3e-4)
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    weight_decay = cfg.get("weight_decay", 0.01)
    eps = cfg.get("eps", 1e-8)

    # All trainable params in one group for now
    # The controller handles selective updates via gradient masking
    params = [p for p in model.parameters() if p.requires_grad]

    return MaskedAdamW(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )

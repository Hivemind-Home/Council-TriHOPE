"""Regularization terms for the distillation objective.

Implements:
- Weight decay: ||θ||²₂
- Trust-region KL: KL(p_θ_old || p_θ)
- Anti-forgetting: Σ ω_i (θ_i - θ_i^old)²
- LoRA sparsity: ||φ_F||₁
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RegularizationConfig:
    """Regularization configuration."""

    weight_decay: float = 0.01
    trust_region_weight: float = 0.0
    anti_forgetting_weight: float = 0.0
    lora_sparsity_weight: float = 0.0


def compute_weight_decay(params: Iterable[nn.Parameter]) -> torch.Tensor:
    """L2 weight decay: ||θ||²₂."""
    total = torch.tensor(0.0)
    for p in params:
        if p.requires_grad:
            total = total + p.float().pow(2).sum()
    return total


def compute_trust_region_kl(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> torch.Tensor:
    """Trust-region KL divergence: KL(p_θ_old || p_θ_new).

    Prevents the student from drifting too far from its previous behavior.

    Args:
        old_logits: [B, T, V] logits from previous parameters (detached).
        new_logits: [B, T, V] logits from current parameters.

    Returns:
        Scalar KL divergence.
    """
    p_old = F.softmax(old_logits.detach(), dim=-1)
    log_p_new = F.log_softmax(new_logits, dim=-1)
    return F.kl_div(log_p_new, p_old, reduction="batchmean")


def compute_anti_forgetting(
    current_params: dict[str, torch.Tensor],
    anchor_params: dict[str, torch.Tensor],
    importance: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Anti-forgetting penalty: Σ ω_i (θ_i - θ_i^old)².

    Discourages updates that damage previously learned knowledge.

    Args:
        current_params: {name: tensor} current parameter values.
        anchor_params: {name: tensor} previous trusted parameter values.
        importance: {name: tensor} per-parameter importance weights (optional).

    Returns:
        Scalar penalty.
    """
    total = torch.tensor(0.0)
    for name, current in current_params.items():
        if name in anchor_params:
            diff = (current - anchor_params[name]).pow(2)
            if importance is not None and name in importance:
                diff = diff * importance[name]
            total = total + diff.sum()
    return total


def compute_lora_sparsity(lora_params: Iterable[nn.Parameter]) -> torch.Tensor:
    """L1 sparsity penalty on LoRA parameters: ||φ_F||₁."""
    total = torch.tensor(0.0)
    for p in lora_params:
        total = total + p.float().abs().sum()
    return total


def compute_total_regularization(
    model: nn.Module,
    config: RegularizationConfig,
    old_logits: torch.Tensor | None = None,
    new_logits: torch.Tensor | None = None,
    anchor_params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute combined regularization loss.

    L_reg = λ_wd * ||θ||² + λ_tr * KL(old||new) + λ_af * AF + λ_sp * ||φ_F||₁
    """
    from .student.lora import LoRALinear

    total = torch.tensor(0.0, device=next(model.parameters()).device)

    # Weight decay (applied to all trainable params)
    if config.weight_decay > 0:
        total = total + config.weight_decay * compute_weight_decay(model.parameters())

    # Trust-region KL
    if config.trust_region_weight > 0 and old_logits is not None and new_logits is not None:
        total = total + config.trust_region_weight * compute_trust_region_kl(
            old_logits, new_logits
        )

    # Anti-forgetting
    if config.anti_forgetting_weight > 0 and anchor_params is not None:
        current = {n: p for n, p in model.named_parameters() if p.requires_grad}
        total = total + config.anti_forgetting_weight * compute_anti_forgetting(
            current, anchor_params
        )

    # LoRA sparsity
    if config.lora_sparsity_weight > 0:
        lora_params = []
        for m in model.modules():
            if isinstance(m, LoRALinear):
                lora_params.extend(m.lora_params)
        if lora_params:
            total = total + config.lora_sparsity_weight * compute_lora_sparsity(lora_params)

    return total

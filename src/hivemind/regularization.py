"""Regularization terms for the distillation objective.

Implements every form mentioned in Theory 201 §5:

* Weight decay ``||θ||²₂``
* Trust-region KL ``KL(p_θ_old ‖ p_θ)``
* Anti-forgetting ``Σ ω_i (θ_i − θ_i^old)²``
* LoRA sparsity ``||φ_F||₁``
* Consistency penalty ``D(p_θ(x), p_θ(x̃))`` — KL between two stochastic
  forward passes (dropout / noise) on the same input. Closes the §5.2
  gap.

All consumers reach into LoRA via the ``LoRAAdapter`` protocol so this
file works for both the in-repo backbone and PEFT/Unsloth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

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
    consistency_weight: float = 0.0


def compute_weight_decay(params: Iterable[nn.Parameter]) -> torch.Tensor:
    """L2 weight decay: ``||θ||²₂``.

    Note: prefer the optimizer's decoupled ``weight_decay`` (AdamW) over
    adding this as a loss term — it avoids materializing a fp32 copy of
    every parameter (~16 GB peak for a 4B model). Kept here for ablations
    that explicitly want the term in the loss.
    """
    total = torch.zeros((), dtype=torch.float32)
    for p in params:
        if not p.requires_grad:
            continue
        # Accumulate in native dtype, only the scalar reduction is upcast,
        # so peak temp memory is one ``p.pow(2)`` (same dtype/size as p)
        # instead of a full fp32 copy.
        s = p.pow(2).sum()
        if total.device != s.device:
            total = total.to(s.device)
        total = total + s.to(torch.float32)
    return total


def compute_trust_region_kl(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> torch.Tensor:
    """``KL(p_θ_old ‖ p_θ_new)`` — keeps updates close to prior behavior."""
    p_old = F.softmax(old_logits.detach(), dim=-1)
    log_p_new = F.log_softmax(new_logits, dim=-1)
    return F.kl_div(log_p_new, p_old, reduction="batchmean")


def compute_anti_forgetting(
    current_params: dict[str, torch.Tensor],
    anchor_params: dict[str, torch.Tensor],
    importance: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``Σ ω_i (θ_i − θ_i^old)²``."""
    total = torch.tensor(0.0)
    for name, current in current_params.items():
        if name in anchor_params:
            diff = (current - anchor_params[name]).pow(2)
            if importance is not None and name in importance:
                diff = diff * importance[name]
            total = total + diff.sum()
    return total


def compute_lora_sparsity(lora_params: Iterable[nn.Parameter]) -> torch.Tensor:
    """L1 sparsity penalty on LoRA parameters: ``||φ_F||₁``."""
    total = torch.tensor(0.0)
    for p in lora_params:
        total = total + p.float().abs().sum()
    return total


def compute_consistency_penalty(
    forward_fn: Callable[[], torch.Tensor],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Two-pass dropout consistency: ``KL(p₁.detach() ‖ p₂)``.

    Theory 201 §5.2: the student should behave consistently under small
    perturbations. We use dropout sampling — the simplest perturbation
    that doesn't require building a second token sequence. The caller
    passes a no-arg callable that runs one forward pass; we call it
    twice. Both passes must keep dropout active (i.e. caller must set
    ``model.train()``).

    When ``attention_mask`` and/or ``labels`` are given we restrict the
    KL to "real" positions (mask=1, label != ignore) so prompt and pad
    tokens don't dilute the signal.
    """
    logits_1 = forward_fn().detach()
    logits_2 = forward_fn()

    p1 = F.softmax(logits_1, dim=-1)
    log_p2 = F.log_softmax(logits_2, dim=-1)

    # [B, T] elementwise KL
    pointwise = (p1 * (torch.log(p1.clamp_min(1e-12)) - log_p2)).sum(dim=-1)

    mask: Optional[torch.Tensor] = None
    if labels is not None:
        mask = (labels != ignore_index).to(pointwise.dtype)
    elif attention_mask is not None:
        mask = attention_mask.to(pointwise.dtype)

    if mask is not None:
        pointwise = pointwise * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(pointwise.numel()), device=pointwise.device).clamp_min(1.0)

    return pointwise.sum() / denom


def compute_total_regularization(
    model: nn.Module,
    config: RegularizationConfig,
    old_logits: torch.Tensor | None = None,
    new_logits: torch.Tensor | None = None,
    anchor_params: dict[str, torch.Tensor] | None = None,
    consistency_forward: Callable[[], torch.Tensor] | None = None,
    consistency_attention_mask: torch.Tensor | None = None,
    consistency_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine every regularization term enabled by ``config``."""
    from .student.lora_adapter import iter_lora_adapters

    total = torch.tensor(0.0, device=next(model.parameters()).device)

    if config.weight_decay > 0:
        total = total + config.weight_decay * compute_weight_decay(model.parameters())

    if config.trust_region_weight > 0 and old_logits is not None and new_logits is not None:
        total = total + config.trust_region_weight * compute_trust_region_kl(
            old_logits, new_logits
        )

    if config.anti_forgetting_weight > 0 and anchor_params is not None:
        current = {n: p for n, p in model.named_parameters() if p.requires_grad}
        total = total + config.anti_forgetting_weight * compute_anti_forgetting(
            current, anchor_params
        )

    if config.lora_sparsity_weight > 0:
        lora_params: list[nn.Parameter] = []
        for _name, adapter in iter_lora_adapters(model):
            lora_params.extend(adapter.lora_params)
        if lora_params:
            total = total + config.lora_sparsity_weight * compute_lora_sparsity(lora_params)

    if config.consistency_weight > 0 and consistency_forward is not None:
        total = total + config.consistency_weight * compute_consistency_penalty(
            consistency_forward,
            attention_mask=consistency_attention_mask,
            labels=consistency_labels,
        )

    return total

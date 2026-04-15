"""Distillation loss functions.

Implements the training objective from Theory 201:
    L(t) = λ_KD · L_KD + λ_CE · L_CE + λ_reg · L_reg

where:
    L_KD = τ² · KL(softmax(z_T/τ) ‖ softmax(z_S/τ))
    L_CE = -Σ log p_θ(y_m | x, y_{<m})  (token-level)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    """Distillation objective configuration."""

    tau: float = 4.0  # temperature
    lambda_kd: float = 1.0
    lambda_ce: float = 0.5
    lambda_reg: float = 0.01


def compute_kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Temperature-scaled KL divergence loss for knowledge distillation.

    L_KD = τ² · KL(p_T^τ ‖ p_S^τ)

    Args:
        teacher_logits: [B, T, V] teacher logit outputs.
        student_logits: [B, T, V] student logit outputs.
        tau: temperature for softening distributions.

    Returns:
        Scalar KD loss.
    """
    p_teacher = F.softmax(teacher_logits / tau, dim=-1)
    log_p_student = F.log_softmax(student_logits / tau, dim=-1)
    # KL(p_T || p_S) = Σ p_T * (log p_T - log p_S)
    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return tau**2 * kl


def compute_ce_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-level cross-entropy loss.

    L_CE = -Σ_m log p_θ(y_{t,m} | x_t, y_{t,<m})

    Args:
        student_logits: [B, T, V] student outputs.
        targets: [B, T] target token indices.
        ignore_index: index to ignore in loss computation.

    Returns:
        Scalar CE loss.
    """
    B, T, V = student_logits.shape
    # Shift: predict next token
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        ignore_index=ignore_index,
    )


def compute_distillation_objective(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    targets: torch.Tensor | None,
    config: DistillationConfig,
    reg_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined distillation objective.

    L(t) = λ_KD · L_KD + λ_CE · L_CE + λ_reg · L_reg

    Args:
        teacher_logits: [B, T, V]
        student_logits: [B, T, V]
        targets: [B, T] or None (if no supervised labels).
        config: loss weighting configuration.
        reg_loss: optional precomputed regularization loss.

    Returns:
        (total_loss, metrics_dict)
    """
    metrics: dict[str, float] = {}

    # KD loss
    kd_loss = compute_kd_loss(teacher_logits, student_logits, config.tau)
    total_loss = config.lambda_kd * kd_loss
    metrics["loss/kd"] = kd_loss.item()

    # CE loss (optional)
    if targets is not None and config.lambda_ce > 0:
        ce_loss = compute_ce_loss(student_logits, targets)
        total_loss = total_loss + config.lambda_ce * ce_loss
        metrics["loss/ce"] = ce_loss.item()

    # Regularization (optional)
    if reg_loss is not None and config.lambda_reg > 0:
        total_loss = total_loss + config.lambda_reg * reg_loss
        metrics["loss/reg"] = reg_loss.item()

    metrics["loss/total"] = total_loss.item()
    return total_loss, metrics

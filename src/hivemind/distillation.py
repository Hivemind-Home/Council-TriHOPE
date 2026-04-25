"""Distillation loss functions.

Implements the Theory 201 training objective:

    L(t) = λ_KD · L_KD + λ_CE · L_CE + λ_reg · L_reg

where

    L_KD = τ² · KL(softmax(z_T/τ) ‖ softmax(z_S/τ))
    L_CE = −Σ log p_θ(y_m | x, y_{<m})  (token-level, shift-by-one)

Two call conventions are supported:

* Synthetic / legacy: ``targets`` is the raw token tensor; next-token CE
  is computed on the full sequence.
* HF mode: ``labels`` is a ``[B, T]`` tensor with ``-100`` on prompt and
  padding positions. CE honours ``ignore_index`` so only response
  positions contribute. A per-row ``teacher_logits_mask`` turns KD on
  or off per sample — rows with missing NPZ cache get zero KD but still
  get CE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class DistillationConfig:
    """Distillation objective configuration."""

    tau: float = 4.0
    lambda_kd: float = 1.0
    lambda_ce: float = 0.5
    lambda_reg: float = 0.01
    use_teacher_confidence: bool = False
    confidence_floor: float = 0.0
    confidence_power: float = 1.0


def _shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard causal-LM shift: predict position t+1 from position t."""
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def compute_kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    tau: float,
    active_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Temperature-scaled KL divergence: τ² · KL(p_T ‖ p_S).

    Args:
        teacher_logits: ``[B, T, V]``.
        student_logits: ``[B, T, V]``.
        tau: temperature.
        active_mask: optional ``[B, T]`` bool/float — 1 on positions that
            should contribute to KD (e.g. non-prompt, non-pad).
        row_mask: optional ``[B]`` float — 0 disables KD for that row
            (NPZ cache missing).
    """
    t_shift, s_shift = teacher_logits[:, :-1, :], student_logits[:, :-1, :]
    p_teacher = F.softmax(t_shift / tau, dim=-1)
    log_p_student = F.log_softmax(s_shift / tau, dim=-1)
    # Pointwise KL: [B, T-1, V] summed over vocab → [B, T-1]
    pointwise = (p_teacher * (torch.log(p_teacher.clamp_min(1e-12)) - log_p_student)).sum(dim=-1)

    if active_mask is not None:
        m = active_mask[:, 1:].to(pointwise.dtype)
        pointwise = pointwise * m
    if row_mask is not None:
        pointwise = pointwise * row_mask.view(-1, 1).to(pointwise.dtype)

    # Normalize by the count of contributing positions so KD magnitude
    # is comparable across batches with different numbers of cached rows.
    denom_parts: list[torch.Tensor] = []
    if active_mask is not None:
        denom_parts.append(active_mask[:, 1:].to(pointwise.dtype))
    else:
        denom_parts.append(torch.ones_like(pointwise))
    if row_mask is not None:
        denom_parts.append(row_mask.view(-1, 1).to(pointwise.dtype).expand_as(pointwise))
    denom = denom_parts[0]
    for d in denom_parts[1:]:
        denom = denom * d
    n = denom.sum().clamp_min(1.0)

    return tau * tau * pointwise.sum() / n


def compute_ce_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Token-level cross-entropy with standard shift-by-one.

    Works for both conventions: pass raw tokens (everything contributes)
    or a labels tensor with ``-100`` on prompt/pad (only response
    positions contribute).
    """
    B, T, V = student_logits.shape
    shift_logits, shift_targets = _shift(student_logits, targets)
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
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    teacher_logits_mask: torch.Tensor | None = None,
    teacher_confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined distillation objective.

    ``L = λ_kd · L_KD + λ_ce · L_CE + λ_reg · L_reg``

    When both ``labels`` and ``targets`` are provided, ``labels`` wins —
    it carries the ``-100`` mask that confines CE to response tokens.

    ``teacher_logits_mask`` is a ``[B]`` float mask that disables KD for
    rows where the teacher NPZ cache missed. The remaining rows still
    contribute to CE via ``labels``.

    ``teacher_confidence`` is an optional ``[B]`` float tensor from Layer
    B. When ``config.use_teacher_confidence`` is True, it multiplies the
    per-row KD mask so low-confidence teacher outputs contribute less —
    see Theory 201 §1 "teacher confidence/entropy routing cues".
    """
    metrics: dict[str, float] = {}

    # CE
    ce_targets = labels if labels is not None else targets
    ce_loss: torch.Tensor | None = None
    if ce_targets is not None and config.lambda_ce > 0:
        ce_loss = compute_ce_loss(student_logits, ce_targets)

    # KD — active positions are "where CE contributes", i.e. response tokens
    kd_active_mask: torch.Tensor | None = None
    if labels is not None:
        kd_active_mask = (labels != IGNORE_INDEX).to(student_logits.dtype)
    elif attention_mask is not None:
        kd_active_mask = attention_mask.to(student_logits.dtype)

    effective_row_mask = teacher_logits_mask
    if (
        config.use_teacher_confidence
        and teacher_confidence is not None
        and teacher_logits_mask is not None
    ):
        conf = teacher_confidence.to(teacher_logits_mask.dtype).clamp_min(config.confidence_floor)
        if config.confidence_power != 1.0:
            conf = conf.clamp_min(1e-6).pow(config.confidence_power)
        effective_row_mask = teacher_logits_mask * conf

    kd_loss = compute_kd_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        tau=config.tau,
        active_mask=kd_active_mask,
        row_mask=effective_row_mask,
    )

    total = config.lambda_kd * kd_loss
    metrics["loss/kd"] = kd_loss.item()

    if ce_loss is not None:
        total = total + config.lambda_ce * ce_loss
        metrics["loss/ce"] = ce_loss.item()

    if reg_loss is not None and config.lambda_reg > 0:
        total = total + config.lambda_reg * reg_loss
        metrics["loss/reg"] = reg_loss.item()

    metrics["loss/total"] = total.item()
    if teacher_logits_mask is not None:
        metrics["loss/kd_row_hit_frac"] = float(teacher_logits_mask.mean().item())
    if config.use_teacher_confidence and teacher_confidence is not None:
        metrics["loss/teacher_conf_mean"] = float(teacher_confidence.mean().item())
    return total, metrics

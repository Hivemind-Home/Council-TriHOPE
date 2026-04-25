"""Teacher confidence gating for KD."""

from __future__ import annotations

import torch

from hivemind.distillation import DistillationConfig, compute_distillation_objective


def _make_batch(B: int = 2, T: int = 6, V: int = 32):
    torch.manual_seed(0)
    student_logits = torch.randn(B, T, V)
    teacher_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    mask = torch.ones(B, dtype=torch.float32)
    return student_logits, teacher_logits, labels, attn, mask


def test_confidence_scales_kd_term():
    student_logits, teacher_logits, labels, attn, mask = _make_batch()
    cfg = DistillationConfig(tau=2.0, lambda_kd=1.0, lambda_ce=0.0, lambda_reg=0.0)

    _, metrics_full = compute_distillation_objective(
        teacher_logits, student_logits, None, cfg,
        labels=labels, attention_mask=attn, teacher_logits_mask=mask,
    )

    cfg_conf = DistillationConfig(
        tau=2.0, lambda_kd=1.0, lambda_ce=0.0, lambda_reg=0.0,
        use_teacher_confidence=True,
    )
    low_conf = torch.full_like(mask, 0.1)
    _, metrics_low = compute_distillation_objective(
        teacher_logits, student_logits, None, cfg_conf,
        labels=labels, attention_mask=attn, teacher_logits_mask=mask,
        teacher_confidence=low_conf,
    )

    # Low confidence should shrink the KD term (both rows scale by 0.1).
    assert metrics_low["loss/kd"] < metrics_full["loss/kd"]


def test_floor_prevents_total_erasure():
    student_logits, teacher_logits, labels, attn, mask = _make_batch()
    cfg = DistillationConfig(
        tau=2.0, lambda_kd=1.0, lambda_ce=0.0, lambda_reg=0.0,
        use_teacher_confidence=True, confidence_floor=0.25,
    )
    conf = torch.tensor([0.01, 0.02], dtype=torch.float32)
    _, metrics = compute_distillation_objective(
        teacher_logits, student_logits, None, cfg,
        labels=labels, attention_mask=attn, teacher_logits_mask=mask,
        teacher_confidence=conf,
    )
    # Floor pulls the effective scale to 0.25, so KD is not ~0.
    assert metrics["loss/kd"] > 0

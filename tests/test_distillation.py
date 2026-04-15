"""Tests for distillation loss functions."""

import torch

from hivemind.distillation import (
    DistillationConfig,
    compute_ce_loss,
    compute_distillation_objective,
    compute_kd_loss,
)


def test_kd_loss_nonnegative():
    """KD loss is non-negative."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 8, 64)
    student_logits = torch.randn(2, 8, 64)

    loss = compute_kd_loss(teacher_logits, student_logits, tau=4.0)
    assert loss.item() >= 0


def test_kd_loss_zero_for_same_logits():
    """KD loss is zero when student matches teacher."""
    logits = torch.randn(2, 8, 64)
    loss = compute_kd_loss(logits, logits.clone(), tau=4.0)
    assert loss.item() < 1e-5


def test_kd_loss_tau_scaling():
    """Higher tau produces different (typically larger) loss due to tau² factor."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 8, 64)
    student_logits = torch.randn(2, 8, 64)

    loss_tau1 = compute_kd_loss(teacher_logits, student_logits, tau=1.0)
    loss_tau4 = compute_kd_loss(teacher_logits, student_logits, tau=4.0)

    # tau=4 should have tau²=16 scaling factor
    # The actual KL changes with tau, so just check they differ
    assert loss_tau1.item() != loss_tau4.item()


def test_ce_loss_shape():
    """CE loss is a scalar."""
    torch.manual_seed(42)
    student_logits = torch.randn(2, 8, 64)
    targets = torch.randint(0, 64, (2, 8))

    loss = compute_ce_loss(student_logits, targets)
    assert loss.dim() == 0  # scalar


def test_combined_objective():
    """Combined objective includes all components."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 8, 64)
    student_logits = torch.randn(2, 8, 64)
    targets = torch.randint(0, 64, (2, 8))
    reg_loss = torch.tensor(0.5)

    config = DistillationConfig(tau=4.0, lambda_kd=1.0, lambda_ce=0.5, lambda_reg=0.01)

    total_loss, metrics = compute_distillation_objective(
        teacher_logits, student_logits, targets, config, reg_loss
    )

    assert total_loss.dim() == 0
    assert "loss/kd" in metrics
    assert "loss/ce" in metrics
    assert "loss/reg" in metrics
    assert "loss/total" in metrics
    assert total_loss.item() > 0


def test_combined_objective_no_labels():
    """Combined objective works without CE labels."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 8, 64)
    student_logits = torch.randn(2, 8, 64)

    config = DistillationConfig(tau=4.0, lambda_kd=1.0, lambda_ce=0.0, lambda_reg=0.0)

    total_loss, metrics = compute_distillation_objective(
        teacher_logits, student_logits, None, config, None
    )

    assert "loss/kd" in metrics
    assert "loss/ce" not in metrics

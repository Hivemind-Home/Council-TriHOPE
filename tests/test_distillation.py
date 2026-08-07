"""Tests for distillation loss functions."""

import pytest
import torch

from hivemind.distillation import (
    IGNORE_INDEX,
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


class TestCEConfidenceWeighting:
    """In cache mode this is the ONLY route by which confidence reaches the loss.

    ``use_teacher_confidence`` scales the per-row KD mask, but without a
    teacher logits cache that mask is all zeros, so it multiplies nothing.
    """

    @staticmethod
    def _fixture(B=4, T=6, V=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        student = torch.randn(B, T, V, generator=g)
        labels = torch.randint(0, V, (B, T), generator=g)
        labels[:, 0] = IGNORE_INDEX  # prompt position
        teacher = torch.zeros(B, T, V)
        return student, labels, teacher

    def test_unweighted_matches_plain_cross_entropy(self):
        student, labels, _ = self._fixture()
        expected = compute_ce_loss(student, labels)
        got = compute_ce_loss(student, labels, row_weights=None)
        assert torch.allclose(expected, got)

    def test_all_ones_weights_match_the_unweighted_loss(self):
        """The weighted path must be a strict generalization."""
        student, labels, _ = self._fixture()
        B = student.shape[0]
        expected = compute_ce_loss(student, labels)
        got = compute_ce_loss(student, labels, row_weights=torch.ones(B))
        assert torch.allclose(expected, got, atol=1e-6)

    def test_weights_shrink_the_loss_rather_than_renormalizing(self):
        student, labels, _ = self._fixture()
        B = student.shape[0]
        full = compute_ce_loss(student, labels, row_weights=torch.ones(B))
        half = compute_ce_loss(student, labels, row_weights=torch.full((B,), 0.5))
        assert torch.allclose(half, full * 0.5, atol=1e-6)

    def test_a_zero_weight_row_drops_out(self):
        student, labels, _ = self._fixture()
        B = student.shape[0]
        w = torch.ones(B)
        w[0] = 0.0
        weighted = compute_ce_loss(student, labels, row_weights=w)
        assert weighted < compute_ce_loss(student, labels, row_weights=torch.ones(B))

    def test_flag_off_ignores_confidence(self):
        student, labels, teacher = self._fixture()
        B = student.shape[0]
        conf = torch.full((B,), 0.5)
        cfg = DistillationConfig(
            lambda_kd=0.0, lambda_ce=1.0, ce_confidence_weighting=False
        )
        _, m_off = compute_distillation_objective(
            teacher, student, None, cfg, labels=labels, teacher_confidence=conf
        )
        _, m_none = compute_distillation_objective(
            teacher, student, None, cfg, labels=labels, teacher_confidence=None
        )
        assert m_off["loss/ce"] == pytest.approx(m_none["loss/ce"])

    def test_flag_on_scales_ce(self):
        student, labels, teacher = self._fixture()
        B = student.shape[0]
        conf = torch.full((B,), 0.5)
        base = DistillationConfig(lambda_kd=0.0, lambda_ce=1.0)
        on = DistillationConfig(
            lambda_kd=0.0, lambda_ce=1.0, ce_confidence_weighting=True
        )
        _, m_base = compute_distillation_objective(
            teacher, student, None, base, labels=labels, teacher_confidence=conf
        )
        _, m_on = compute_distillation_objective(
            teacher, student, None, on, labels=labels, teacher_confidence=conf
        )
        assert m_on["loss/ce"] == pytest.approx(m_base["loss/ce"] * 0.5, rel=1e-5)

    def test_confidence_floor_applies_to_ce(self):
        student, labels, teacher = self._fixture()
        B = student.shape[0]
        cfg = DistillationConfig(
            lambda_kd=0.0, lambda_ce=1.0, ce_confidence_weighting=True,
            confidence_floor=0.5,
        )
        _, floored = compute_distillation_objective(
            teacher, student, None, cfg,
            labels=labels, teacher_confidence=torch.zeros(B),
        )
        _, at_floor = compute_distillation_objective(
            teacher, student, None, cfg,
            labels=labels, teacher_confidence=torch.full((B,), 0.5),
        )
        assert floored["loss/ce"] == pytest.approx(at_floor["loss/ce"])

    def test_gradients_flow_through_the_weighted_path(self):
        student, labels, _ = self._fixture()
        student.requires_grad_(True)
        loss = compute_ce_loss(student, labels, row_weights=torch.full((4,), 0.7))
        loss.backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()

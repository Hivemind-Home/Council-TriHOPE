"""Tests for stability signals."""

import torch

from hivemind.controller.stability import StabilityTracker
from hivemind.controller.config import StabilityConfig


def test_directional_parallel_vectors():
    """C_t = 1 for parallel vectors."""
    tracker = StabilityTracker()
    grad = torch.tensor([1.0, 2.0, 3.0])
    m = torch.tensor([2.0, 4.0, 6.0])  # same direction
    c = tracker.compute_directional(grad, m)
    assert abs(c - 1.0) < 1e-5


def test_directional_anti_parallel():
    """C_t = -1 for anti-parallel vectors."""
    tracker = StabilityTracker()
    grad = torch.tensor([1.0, 2.0, 3.0])
    m = torch.tensor([-1.0, -2.0, -3.0])
    c = tracker.compute_directional(grad, m)
    assert abs(c - (-1.0)) < 1e-5


def test_directional_orthogonal():
    """C_t ≈ 0 for orthogonal vectors."""
    tracker = StabilityTracker()
    grad = torch.tensor([1.0, 0.0])
    m = torch.tensor([0.0, 1.0])
    c = tracker.compute_directional(grad, m)
    assert abs(c) < 1e-5


def test_directional_warmup_guard():
    """C_t = 0 when m is too small (warmup)."""
    tracker = StabilityTracker(StabilityConfig(warmup_threshold=1.0))
    grad = torch.tensor([1.0, 2.0])
    m = torch.tensor([1e-8, 1e-8])  # below threshold
    c = tracker.compute_directional(grad, m)
    assert c == 0.0


def test_adam_ratio_perfect_consistency():
    """Adam ratio ≈ 1 when m² ≈ v (all signal, no noise)."""
    tracker = StabilityTracker()
    m = torch.tensor([1.0, 2.0, 3.0])
    v = m.pow(2)  # v = m², so ratio = m²/(m²+ε) ≈ 1
    stab = tracker.compute_adam_ratio(m, v)
    assert stab > 0.99


def test_adam_ratio_pure_noise():
    """Adam ratio ≈ 0 when m ≈ 0 but v is large (all noise)."""
    tracker = StabilityTracker()
    m = torch.tensor([0.001, -0.001, 0.001])
    v = torch.tensor([10.0, 10.0, 10.0])
    stab = tracker.compute_adam_ratio(m, v)
    assert stab < 0.01


def test_windowed_variance_constant_norm():
    """V_t ≈ 0 for constant gradient norms."""
    tracker = StabilityTracker(StabilityConfig(ema_alpha=0.3))

    # Feed same norm repeatedly
    for _ in range(50):
        v = tracker.update_windowed_variance(5.0)

    # Variance should be near zero for constant input
    assert v < 0.01


def test_windowed_variance_volatile():
    """V_t > 0 for oscillating gradient norms."""
    tracker = StabilityTracker(StabilityConfig(ema_alpha=0.3))

    # Alternate between small and large norms
    for i in range(50):
        norm = 1.0 if i % 2 == 0 else 10.0
        v = tracker.update_windowed_variance(norm)

    # Should show significant variance
    assert v > 0.1

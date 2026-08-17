"""Tests for stability signals."""

import torch

from hivemind.controller.config import StabilityConfig
from hivemind.controller.stability import StabilityTracker


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


def test_sustained_C_seeds_then_emas():
    """C̄ seeds to the first non-warmup cosine, then EMAs subsequent ones."""
    tracker = StabilityTracker(StabilityConfig(c_ema_alpha=0.1))
    m = torch.tensor([1.0, 0.0])  # momentum along x

    # First observation: gradient parallel to m → cos = 1 → seeds C̄ = 1.0
    tracker.compute_directional(torch.tensor([2.0, 0.0]), m)
    assert abs(tracker.sustained_C - 1.0) < 1e-6

    # Second: orthogonal gradient → cos = 0 → C̄ = 0.9*1.0 + 0.1*0
    tracker.compute_directional(torch.tensor([0.0, 1.0]), m)
    assert abs(tracker.sustained_C - 0.9) < 1e-6

    # Third: orthogonal again → C̄ = 0.9*0.9
    tracker.compute_directional(torch.tensor([0.0, 1.0]), m)
    assert abs(tracker.sustained_C - 0.81) < 1e-6


def test_sustained_C_smooths_over_a_noisy_instant():
    """A single low-cosine step barely dents a long high-cosine history.

    This is the property that makes ``stability_mode="sustained"`` fire on
    genuinely-stable modules: the instantaneous cosine at one step can be
    low, yet C̄ stays high.
    """
    tracker = StabilityTracker(StabilityConfig(c_ema_alpha=0.1))
    m = torch.tensor([1.0, 0.0])

    for _ in range(50):  # long stretch of perfect alignment
        tracker.compute_directional(torch.tensor([1.0, 0.0]), m)
    assert tracker.sustained_C > 0.99

    # One noisy (anti-aligned) step: instantaneous cosine = -1 ...
    c_instant = tracker.compute_directional(torch.tensor([-1.0, 0.0]), m)
    assert abs(c_instant - (-1.0)) < 1e-6
    # ... but the sustained signal is barely moved (still clears a 0.7 bar).
    assert tracker.sustained_C > 0.7


def test_sustained_C_ignores_warmup_steps():
    """Warmup steps (tiny ‖m‖) do not seed or move C̄."""
    tracker = StabilityTracker(StabilityConfig(c_ema_alpha=0.1, warmup_threshold=1e-6))
    tiny_m = torch.tensor([1e-9, 0.0])

    tracker.compute_directional(torch.tensor([1.0, 0.0]), tiny_m)
    assert tracker.sustained_C == 0.0  # nothing recorded during warmup

    # A real observation then seeds it.
    tracker.compute_directional(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert abs(tracker.sustained_C - 1.0) < 1e-6


def test_sustained_C_survives_state_dict_roundtrip():
    """C̄ persists across checkpoint save/load; old checkpoints start cold."""
    tracker = StabilityTracker(StabilityConfig(c_ema_alpha=0.1))
    m = torch.tensor([1.0, 0.0])
    for _ in range(10):
        tracker.compute_directional(torch.tensor([1.0, 0.0]), m)
    saved = tracker.state_dict()

    restored = StabilityTracker(StabilityConfig(c_ema_alpha=0.1))
    restored.load_state_dict(saved)
    assert restored.sustained_C == tracker.sustained_C

    # Pre-C̄ checkpoint (missing the keys) loads cold, no crash.
    legacy = StabilityTracker(StabilityConfig())
    legacy.load_state_dict({"_ema_norm": 1.0, "_ema_norm_sq": 1.0, "_initialized": True})
    assert legacy.sustained_C == 0.0

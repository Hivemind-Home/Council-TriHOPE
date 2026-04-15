"""Tests for Adam-based surprise signal."""

import torch

from hivemind.controller.surprise import AdamSurprise
from hivemind.controller.config import SurpriseConfig


def test_surprise_known_values():
    """Surprise matches hand-computed value for known inputs."""
    config = SurpriseConfig(s_min=0.0, s_max=100.0, eps=0.0)
    surprise = AdamSurprise(config)

    # g = [2, 4], v = [1, 4]
    # S = (1/2) * (4/1 + 16/4) = (1/2) * (4 + 4) = 4.0
    grad = torch.tensor([2.0, 4.0])
    v = torch.tensor([1.0, 4.0])

    s = surprise.compute(grad, v)
    assert abs(s - 4.0) < 1e-5


def test_surprise_clamping():
    """Surprise is clamped to [s_min, s_max]."""
    config = SurpriseConfig(s_min=0.5, s_max=5.0, eps=1e-8)
    surprise = AdamSurprise(config)

    # Very large gradient -> high surprise, should be clamped
    grad = torch.tensor([100.0, 100.0])
    v = torch.tensor([0.01, 0.01])

    s = surprise.compute(grad, v)
    assert s == 5.0

    # Very small gradient -> low surprise, should be clamped
    grad = torch.tensor([0.0, 0.0])
    v = torch.tensor([1.0, 1.0])

    s = surprise.compute(grad, v)
    assert s == 0.5


def test_surprise_typical_range():
    """When grad ~ sqrt(v), surprise should be near 1."""
    config = SurpriseConfig(s_min=0.0, s_max=100.0, eps=1e-8)
    surprise = AdamSurprise(config)

    # g_i² ≈ v_i means each term ≈ 1, so S ≈ 1
    v = torch.tensor([1.0, 4.0, 9.0])
    grad = torch.tensor([1.0, 2.0, 3.0])  # g_i = sqrt(v_i)

    s = surprise.compute(grad, v)
    assert abs(s - 1.0) < 0.01

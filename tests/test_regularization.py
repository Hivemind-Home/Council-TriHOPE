"""Tests for regularization terms."""

import torch
import torch.nn as nn

from hivemind.regularization import (
    compute_anti_forgetting,
    compute_lora_sparsity,
    compute_trust_region_kl,
    compute_weight_decay,
)


def test_weight_decay_positive():
    """Weight decay produces a positive scalar."""
    params = [nn.Parameter(torch.randn(10, 10))]
    loss = compute_weight_decay(params)
    assert loss.item() > 0


def test_weight_decay_zero_for_zero_params():
    """Weight decay is zero for zero parameters."""
    params = [nn.Parameter(torch.zeros(10, 10))]
    loss = compute_weight_decay(params)
    assert loss.item() == 0.0


def test_trust_region_kl_zero_for_same():
    """Trust region KL is zero when old == new."""
    logits = torch.randn(2, 8, 64)
    loss = compute_trust_region_kl(logits, logits.clone())
    assert loss.item() < 1e-5


def test_trust_region_kl_nonnegative():
    """Trust region KL is non-negative."""
    torch.manual_seed(42)
    old = torch.randn(2, 8, 64)
    new = torch.randn(2, 8, 64)
    loss = compute_trust_region_kl(old, new)
    assert loss.item() >= 0


def test_anti_forgetting():
    """Anti-forgetting penalty penalizes parameter drift."""
    current = {"w": torch.tensor([1.0, 2.0, 3.0])}
    anchor = {"w": torch.tensor([1.0, 1.0, 1.0])}
    loss = compute_anti_forgetting(current, anchor)
    # (0² + 1² + 2²) = 5
    assert abs(loss.item() - 5.0) < 1e-5


def test_anti_forgetting_with_importance():
    """Anti-forgetting uses importance weights."""
    current = {"w": torch.tensor([1.0, 2.0, 3.0])}
    anchor = {"w": torch.tensor([1.0, 1.0, 1.0])}
    importance = {"w": torch.tensor([0.0, 1.0, 2.0])}
    loss = compute_anti_forgetting(current, anchor, importance)
    # 0*0² + 1*1² + 2*2² = 0 + 1 + 8 = 9
    assert abs(loss.item() - 9.0) < 1e-5


def test_lora_sparsity():
    """LoRA sparsity is L1 norm."""
    params = [nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))]
    loss = compute_lora_sparsity(params)
    assert abs(loss.item() - 6.0) < 1e-5

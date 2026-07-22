"""Tests for LoRA adapter module."""

import torch
import torch.nn as nn

from hivemind.student.lora import LoRALinear


def test_lora_forward_shape():
    """LoRA forward produces correct output shape."""
    torch.manual_seed(42)
    lora = LoRALinear(32, 64, rank=4, alpha=8.0)
    x = torch.randn(2, 16, 32)
    out = lora(x)
    assert out.shape == (2, 16, 64)


def test_lora_forward_includes_base_and_delta():
    """LoRA output = base + delta, not just base."""
    torch.manual_seed(42)
    lora = LoRALinear(32, 64, rank=4, alpha=8.0)
    x = torch.randn(2, 16, 32)

    base_out = lora.base(x)
    _ = lora(x)

    # After init, B is zeros so delta should be ~0
    # But A is Kaiming, so let's set B nonzero
    nn.init.normal_(lora.lora_B, std=0.1)
    full_out_nonzero = lora(x)

    # Now they should differ
    assert not torch.allclose(base_out, full_out_nonzero, atol=1e-6)


def test_lora_merge_changes_base():
    """Merging LoRA into base changes base weights."""
    torch.manual_seed(42)
    lora = LoRALinear(32, 64, rank=4, alpha=8.0)
    nn.init.normal_(lora.lora_B, std=0.1)

    base_before = lora.base.weight.data.clone()
    lora.merge_lora_into_base()
    base_after = lora.base.weight.data

    assert not torch.allclose(base_before, base_after)


def test_lora_reset_zeros_adapter():
    """Reset LoRA zeros out B matrix."""
    torch.manual_seed(42)
    lora = LoRALinear(32, 64, rank=4, alpha=8.0)
    nn.init.normal_(lora.lora_B, std=0.1)

    lora.reset_lora()
    assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))


def test_lora_param_groups_separation():
    """Base and LoRA params are correctly separated."""
    lora = LoRALinear(32, 64, rank=4, alpha=8.0)

    base_params = lora.base_params
    lora_params = lora.lora_params

    assert len(base_params) == 1  # weight only (no bias)
    assert len(lora_params) == 2  # A and B
    assert base_params[0].shape == (64, 32)
    assert lora_params[0].shape == (4, 32)  # A: rank x in
    assert lora_params[1].shape == (64, 4)  # B: out x rank


def test_lora_top_k_mask_rank_components():
    """Top-K mask selects correct number of rank components."""
    torch.manual_seed(42)
    lora = LoRALinear(32, 64, rank=4, alpha=8.0)
    x = torch.randn(2, 8, 32)
    out = lora(x)
    out.sum().backward()

    mask_A, mask_B = lora.get_top_k_mask(k=2, granularity="rank_components")

    # Should have exactly 2 rank components active
    active_ranks_A = mask_A[:, 0].sum().item()
    active_ranks_B = mask_B[0, :].sum().item()
    assert active_ranks_A == 2
    assert active_ranks_B == 2

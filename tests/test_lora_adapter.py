"""LoRAAdapter protocol — coverage for both Native and PEFT shapes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from hivemind.student.lora import LoRALinear
from hivemind.student.lora_adapter import (
    _PEFTAdapter,
    get_adapter,
    iter_lora_adapters,
)


def _make_native(rank=4):
    return LoRALinear(in_features=8, out_features=16, rank=rank, alpha=8.0)


def _make_peft_like(rank=4, in_f=8, out_f=16, scale=1.5):
    """Hand-built fake PEFT lora.Linear with the duck-type surface our
    ``_PEFTAdapter`` reads from. Avoids importing peft."""

    base_layer = nn.Linear(in_f, out_f, bias=False)
    lora_A = nn.ModuleDict({"default": nn.Linear(in_f, rank, bias=False)})
    lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_f, bias=False)})
    # B starts at zero (LoRA convention) so ΔW = 0 initially
    with torch.no_grad():
        lora_B["default"].weight.zero_()

    class _Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = base_layer
            self.lora_A = lora_A
            self.lora_B = lora_B
            self.scaling = {"default": scale}

        def forward(self, x):
            base = self.base_layer(x)
            delta = lora_B["default"](lora_A["default"](x)) * scale
            return base + delta

    return _Fake()


def test_native_adapter_roundtrip():
    m = _make_native()
    a = get_adapter(m)
    assert a is not None
    assert a.rank == 4
    assert a.out_features == 16
    assert a.base_weight is m.base.weight
    assert a.lora_a is m.lora_A
    assert a.lora_b is m.lora_B


def test_peft_adapter_roundtrip():
    m = _make_peft_like()
    a = get_adapter(m)
    assert a is not None
    assert isinstance(a, _PEFTAdapter)
    assert a.rank == 4
    assert a.out_features == 16
    assert a.scaling == 1.5


def test_iter_lora_adapters_finds_both_shapes():
    parent = nn.Module()
    parent.add_module("native", _make_native())
    parent.add_module("peft", _make_peft_like())

    found = dict(iter_lora_adapters(parent))
    assert "native" in found and "peft" in found


def test_top_k_mask_native_and_peft_agree_on_shape():
    """Both adapters should produce identically-shaped Top-K masks for
    rank-component granularity. (Values differ because gradients differ.)"""
    rank = 4
    native = _make_native(rank=rank)
    peft = _make_peft_like(rank=rank)

    # Make gradients exist
    x = torch.randn(2, 8)
    native(x).sum().backward()
    peft(x).sum().backward()

    a_native, b_native = get_adapter(native).get_top_k_mask(2, "rank_components")
    a_peft, b_peft = get_adapter(peft).get_top_k_mask(2, "rank_components")

    # Native A is [rank, in_features], B is [out, rank].
    # PEFT lora_A.weight is [rank, in], lora_B.weight is [out, rank].
    assert a_native.shape == a_peft.shape
    assert b_native.shape == b_peft.shape


def test_peft_merge_then_reset_zeros_b():
    m = _make_peft_like(rank=4)
    # Make B non-zero so we can verify it's reset post-merge.
    with torch.no_grad():
        m.lora_B["default"].weight.normal_(std=0.05)

    base_before = m.base_layer.weight.detach().clone()
    a = get_adapter(m)
    a.merge_lora_into_base()
    base_after = m.base_layer.weight

    assert torch.all(m.lora_B["default"].weight == 0)
    # Base must have moved (we wrote a non-zero delta into it)
    assert not torch.allclose(base_before, base_after)


def test_peft_reset_lora_zeros_b_kaiming_a():
    m = _make_peft_like(rank=4)
    a = get_adapter(m)
    # randomize first
    with torch.no_grad():
        m.lora_A["default"].weight.normal_()
        m.lora_B["default"].weight.normal_()
    a.reset_lora()
    assert torch.all(m.lora_B["default"].weight == 0)
    # A is non-zero (kaiming) — just verify it's not all zeros
    assert m.lora_A["default"].weight.abs().sum().item() > 0


def test_unknown_module_returns_none():
    plain = nn.Linear(4, 4)
    assert get_adapter(plain) is None

"""HFStudent LoRA injection against a monkeypatched HF-style model.

Avoids pulling real Qwen weights — we build a minimal torch module with
Qwen-shaped attribute paths and assert that HFStudent's injection
swaps the target ``nn.Linear`` projections to ``LoRALinear`` and
exposes the ``blocks`` view ``build_module_index`` expects.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from hivemind.student.config import LoRAConfig
from hivemind.student.lora import LoRALinear


def _install_fake_transformers(monkeypatch, vocab_size=128, hidden=32, n_layers=2, heads=4):
    """Build a tiny Qwen-like torch stack + stub AutoModelForCausalLM."""

    class _AttnMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

    class _MLPMod(nn.Module):
        def __init__(self):
            super().__init__()
            mid = hidden * 2
            self.gate_proj = nn.Linear(hidden, mid, bias=False)
            self.up_proj = nn.Linear(hidden, mid, bias=False)
            self.down_proj = nn.Linear(mid, hidden, bias=False)

    class _LayerMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _AttnMod()
            self.mlp = _MLPMod()

        def forward(self, x):
            return x

    class _CoreMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden)
            self.layers = nn.ModuleList([_LayerMod() for _ in range(n_layers)])

        def forward(self, x):
            return x

    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _CoreMod()
            self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

        def forward(self, input_ids=None, use_cache=False, return_dict=True):
            h = self.model.embed_tokens(input_ids)
            return _Output(self.lm_head(h))

    class _Config:
        vocab_size = vocab_size
        hidden_size = hidden
        num_hidden_layers = n_layers
        num_attention_heads = heads
        max_position_embeddings = 64

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _Config()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _CausalLM()

    fake = types.ModuleType("transformers")
    fake.AutoConfig = _AutoConfig
    fake.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return _CausalLM


def test_lora_injection_swaps_projections(monkeypatch):
    _install_fake_transformers(monkeypatch)
    from hivemind.student.hf_backbone import HFStudent

    lora = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "k", "v", "o", "gate", "up", "down"])
    student = HFStudent(pretrained_name="fake/qwen", lora_config=lora)

    assert len(student.blocks) == 2
    for block in student.blocks:
        for name in ["w_q", "w_k", "w_v", "w_o"]:
            assert isinstance(getattr(block.attn, name), LoRALinear)
        for name in ["w_gate", "w_up", "w_down"]:
            assert isinstance(getattr(block.ffn, name), LoRALinear)


def test_forward_returns_logits(monkeypatch):
    _install_fake_transformers(monkeypatch, vocab_size=64, hidden=16, n_layers=2)
    from hivemind.student.hf_backbone import HFStudent

    lora = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "v"])
    student = HFStudent(pretrained_name="fake/qwen", lora_config=lora)
    tokens = torch.randint(0, 64, (2, 8))
    logits = student(tokens)
    assert logits.shape == (2, 8, 64)


def test_param_groups_split(monkeypatch):
    _install_fake_transformers(monkeypatch)
    from hivemind.student.hf_backbone import HFStudent

    lora = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "k"])
    student = HFStudent(pretrained_name="fake/qwen", lora_config=lora)
    groups = student.get_param_groups()
    assert len(groups["F"]) > 0
    assert len(groups["P"]) > 0
    # Every param appears in exactly one group
    ids = {id(p) for p in groups["F"]} | {id(p) for p in groups["P"]} | {id(p) for p in groups["shared"]}
    all_ids = {id(p) for p in student.parameters()}
    assert ids == all_ids


def test_module_index_enumerates_hf_blocks(monkeypatch):
    _install_fake_transformers(monkeypatch)
    from hivemind.controller.module_index import build_module_index
    from hivemind.student.hf_backbone import HFStudent

    lora = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "k", "v", "o", "gate", "up", "down"])
    student = HFStudent(pretrained_name="fake/qwen", lora_config=lora)
    modules = build_module_index(student)
    # 2 layers × {attn, ffn} × {P, F} = 8 modules
    assert len(modules) == 8


def test_unknown_architecture_raises(monkeypatch):
    class _Empty(nn.Module):
        pass

    fake = types.ModuleType("transformers")

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*a, **kw):
            class _C:
                vocab_size = 10
                hidden_size = 8
                num_hidden_layers = 1
                num_attention_heads = 2
                max_position_embeddings = 16
            return _C()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _Empty()

    fake.AutoConfig = _AutoConfig
    fake.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake)

    from hivemind.student.hf_backbone import HFStudent

    with pytest.raises(NotImplementedError):
        HFStudent(pretrained_name="fake/x", lora_config=LoRAConfig(rank=4, target_modules=["q"]))

"""UnslothStudent — uses a fake ``unsloth`` module so no real install needed.

The fake's ``FastLanguageModel.from_pretrained`` returns a Qwen-shaped
torch model, and ``get_peft_model`` wraps the targeted ``nn.Linear``
projections with PEFT-shaped adapters (``base_layer`` + ``lora_A`` /
``lora_B`` ``ModuleDict``s). The point is to exercise the wiring:
detect_arch, ProjView, LoRAAdapter, and the StudentModel-compatible
surface. Real Unsloth lives on the GPU box.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from hivemind.student.config import LoRAConfig
from hivemind.student.lora_adapter import _PEFTAdapter, get_adapter


def _peft_wrap(linear: nn.Linear, rank: int) -> nn.Module:
    in_f, out_f = linear.in_features, linear.out_features

    class _PeftLin(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = linear
            self.lora_A = nn.ModuleDict({"default": nn.Linear(in_f, rank, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_f, bias=False)})
            with torch.no_grad():
                self.lora_B["default"].weight.zero_()
            self.scaling = {"default": 1.0}

        def forward(self, x):
            return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))

    return _PeftLin()


def _install_fake_unsloth(monkeypatch, vocab=128, hidden=32, n_layers=2, heads=4):
    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            mid = hidden * 2
            self.gate_proj = nn.Linear(hidden, mid, bias=False)
            self.up_proj = nn.Linear(hidden, mid, bias=False)
            self.down_proj = nn.Linear(mid, hidden, bias=False)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attn()
            self.mlp = _MLP()

    class _Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])

    class _Cfg:
        vocab_size = vocab
        hidden_size = hidden
        num_hidden_layers = n_layers
        num_attention_heads = heads
        max_position_embeddings = 64

    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Core()
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = _Cfg()

        def forward(self, input_ids=None, use_cache=False, return_dict=True):
            h = self.model.embed_tokens(input_ids)
            return _Output(self.lm_head(h))

    class _PeftModel(nn.Module):
        """Mimics PEFT's wrapping of the causal LM."""

        def __init__(self, base):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = base

        def forward(self, input_ids=None, use_cache=False, return_dict=True):
            return self.base_model.model(
                input_ids=input_ids, use_cache=use_cache, return_dict=return_dict
            )

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _FastLanguageModel:
        @staticmethod
        def from_pretrained(model_name, max_seq_length, dtype, **kw):
            return _CausalLM(), _Tokenizer()

        @staticmethod
        def get_peft_model(model, r, target_modules, **kw):
            for layer in model.model.layers:
                for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if nm in target_modules:
                        setattr(layer.self_attn, nm,
                                _peft_wrap(getattr(layer.self_attn, nm), r))
                for nm in ["gate_proj", "up_proj", "down_proj"]:
                    if nm in target_modules:
                        setattr(layer.mlp, nm, _peft_wrap(getattr(layer.mlp, nm), r))
            return _PeftModel(model)

    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = _FastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", fake)


def test_unsloth_student_builds_and_routes_through_peft(monkeypatch):
    _install_fake_unsloth(monkeypatch, vocab=64, hidden=16, n_layers=2)
    from hivemind.student.unsloth_backbone import UnslothStudent

    lora = LoRAConfig(rank=4, alpha=8.0,
                      target_modules=["q", "k", "v", "o", "gate", "up", "down"])
    student = UnslothStudent(pretrained_name="fake/qwen3.5", lora_config=lora,
                             max_seq_len=32)

    assert len(student.blocks) == 2
    for block in student.blocks:
        for name in ["w_q", "w_k", "w_v", "w_o"]:
            mod = getattr(block.attn, name)
            assert isinstance(get_adapter(mod), _PEFTAdapter)
        for name in ["w_gate", "w_up", "w_down"]:
            mod = getattr(block.ffn, name)
            assert isinstance(get_adapter(mod), _PEFTAdapter)


def test_unsloth_student_forward(monkeypatch):
    _install_fake_unsloth(monkeypatch, vocab=64, hidden=16, n_layers=2)
    from hivemind.student.unsloth_backbone import UnslothStudent

    lora = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "v"])
    student = UnslothStudent(pretrained_name="fake/qwen3.5", lora_config=lora)
    tokens = torch.randint(0, 64, (2, 8))
    out = student(tokens)
    assert out.shape == (2, 8, 64)


def test_module_index_sees_unsloth_loras(monkeypatch):
    _install_fake_unsloth(monkeypatch, vocab=64, hidden=16, n_layers=2)
    from hivemind.controller.module_index import build_module_index
    from hivemind.student.unsloth_backbone import UnslothStudent

    lora = LoRAConfig(rank=4, alpha=8.0,
                      target_modules=["q", "k", "v", "o", "gate", "up", "down"])
    student = UnslothStudent(pretrained_name="fake/qwen3.5", lora_config=lora)

    modules = build_module_index(student)
    # 2 layers × {attn, ffn} × {P, F}
    assert len(modules) == 8


def test_param_groups_split_for_unsloth(monkeypatch):
    _install_fake_unsloth(monkeypatch, vocab=64, hidden=16, n_layers=2)
    from hivemind.student.unsloth_backbone import UnslothStudent

    lora = LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "k"])
    student = UnslothStudent(pretrained_name="fake/qwen3.5", lora_config=lora)
    groups = student.get_param_groups()
    assert len(groups["F"]) > 0
    assert len(groups["P"]) > 0

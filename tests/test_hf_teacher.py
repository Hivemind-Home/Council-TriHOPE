"""HFTeacher smoke + vocab-mismatch guard."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn


def _install_fake_transformers(monkeypatch, vocab_size: int):
    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 8)
            self.head = nn.Linear(8, vocab_size, bias=False)

        def forward(self, input_ids=None, use_cache=False, return_dict=True):
            return _Output(self.head(self.embed(input_ids)))

    class _Config:
        pass
    _Config.vocab_size = vocab_size

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _Config()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*a, **kw):
            return _Model()

    fake = types.ModuleType("transformers")
    fake.AutoConfig = _AutoConfig
    fake.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_forward_returns_logits(monkeypatch):
    _install_fake_transformers(monkeypatch, vocab_size=32)
    from hivemind.teacher_hf import HFTeacher

    teacher = HFTeacher(pretrained_name="fake/x", expected_vocab_size=32)
    tokens = torch.randint(0, 32, (2, 4))
    logits = teacher(tokens)
    assert logits.shape == (2, 4, 32)


def test_vocab_mismatch_raises(monkeypatch):
    _install_fake_transformers(monkeypatch, vocab_size=100)
    from hivemind.teacher_hf import HFTeacher

    with pytest.raises(ValueError, match="vocab mismatch"):
        HFTeacher(pretrained_name="fake/x", expected_vocab_size=200)


def test_parameters_are_frozen(monkeypatch):
    _install_fake_transformers(monkeypatch, vocab_size=16)
    from hivemind.teacher_hf import HFTeacher

    teacher = HFTeacher(pretrained_name="fake/x", expected_vocab_size=16)
    for p in teacher.parameters():
        assert not p.requires_grad


def test_parse_specs_dict_and_list():
    from hivemind.teacher_hf import parse_hf_teacher_specs

    from_dict = parse_hf_teacher_specs({"code": "Qwen/X", "math": "meta/Y"})
    assert {s.domain for s in from_dict} == {"code", "math"}

    from_list = parse_hf_teacher_specs([
        {"domain": "code", "pretrained_name": "Qwen/X", "teacher_id": "code_v1"},
    ])
    assert from_list[0].teacher_id == "code_v1"

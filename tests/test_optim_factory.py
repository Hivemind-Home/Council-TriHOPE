"""Tests for the optimizer factory param-group split."""

from __future__ import annotations

from hivemind.optim.factory import build_optimizer
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _tiny_student() -> StudentModel:
    cfg = StudentConfig(
        vocab_size=64,
        dim=32,
        num_layers=2,
        heads=2,
        max_seq_len=32,
        lora=LoRAConfig(rank=4, alpha=8.0),
    )
    return StudentModel(cfg)


def test_groups_split_by_store() -> None:
    model = _tiny_student()
    opt = build_optimizer(model, {"lr": 1e-3})
    names = {g["name"] for g in opt.param_groups}
    assert names == {"P", "F", "shared"}
    expected = model.get_param_groups()
    for group in opt.param_groups:
        assert len(group["params"]) == len(
            [p for p in expected[group["name"]] if p.requires_grad]
        )


def test_per_group_overrides() -> None:
    model = _tiny_student()
    opt = build_optimizer(
        model,
        {
            "lr": 1e-3,
            "lr_lora": 5e-4,
            "lr_base": 1e-5,
            "weight_decay": 0.01,
            "weight_decay_lora": 0.0,
        },
    )
    by_name = {g["name"]: g for g in opt.param_groups}
    assert by_name["F"]["lr"] == 5e-4
    assert by_name["P"]["lr"] == 1e-5
    assert by_name["shared"]["lr"] == 1e-3
    assert by_name["F"]["weight_decay"] == 0.0
    assert by_name["P"]["weight_decay"] == 0.01


def test_returns_masked_adamw() -> None:
    model = _tiny_student()
    opt = build_optimizer(model, {})
    assert isinstance(opt, MaskedAdamW)

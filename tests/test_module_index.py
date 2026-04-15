"""Tests for module indexing."""

import torch

from hivemind.controller.module_index import ModuleId, build_module_index
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def test_module_count():
    """2 layers with LoRA → 8 modules (attn_P, attn_F, ffn_P, ffn_F per layer)."""
    torch.manual_seed(42)
    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=2, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    model = StudentModel(config)
    modules = build_module_index(model)

    assert len(modules) == 8

    # Check we have both P and F for both attn and ffn in each layer
    ids = {m.id for m in modules}
    for layer in range(2):
        assert ModuleId(layer, "attn", "P") in ids
        assert ModuleId(layer, "attn", "F") in ids
        assert ModuleId(layer, "ffn", "P") in ids
        assert ModuleId(layer, "ffn", "F") in ids


def test_module_params_nonempty():
    """Every module has at least one parameter."""
    torch.manual_seed(42)
    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    model = StudentModel(config)
    modules = build_module_index(model)

    for mod in modules:
        assert mod.num_params > 0, f"Module {mod.id} has no params"


def test_module_id_string():
    """ModuleId has readable string representation."""
    mid = ModuleId(3, "ffn", "F")
    assert str(mid) == "L3.ffn.F"

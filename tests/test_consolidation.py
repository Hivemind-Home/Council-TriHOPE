"""Tests for F→P consolidation."""

import torch

from hivemind.controller.config import ConsolidationConfig
from hivemind.controller.consolidation import ConsolidationScheduler
from hivemind.controller.module_index import ModuleId
from hivemind.controller.signals import ModuleSignals
from hivemind.stores.permanent import PermanentStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.lora import LoRALinear
from hivemind.student.model import StudentModel


def test_consolidation_period():
    """should_check returns True at correct steps."""
    config = ConsolidationConfig(period=100)
    model = StudentModel(StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q"]),
    ))
    scheduler = ConsolidationScheduler(config, model, PermanentStore())

    assert not scheduler.should_check(0)
    assert not scheduler.should_check(50)
    assert scheduler.should_check(100)
    assert scheduler.should_check(200)


def test_consolidation_merges_qualifying_modules():
    """Qualifying F-modules get their LoRA merged into base."""
    torch.manual_seed(42)
    config = ConsolidationConfig(min_stability_C=0.5, min_repetition=0.5, period=100)
    student_cfg = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    model = StudentModel(student_cfg)
    p_store = PermanentStore()
    scheduler = ConsolidationScheduler(config, model, p_store)

    # Set LoRA B to nonzero so merge has an effect
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.1)

    # Create qualifying signals for attn F-module
    mid_qualify = ModuleId(0, "attn", "F")
    mid_no_qualify = ModuleId(0, "ffn", "F")

    signals = {
        mid_qualify: ModuleSignals(
            module_id=mid_qualify, stability_C=0.8, repetition=0.7
        ),
        mid_no_qualify: ModuleSignals(
            module_id=mid_no_qualify, stability_C=0.2, repetition=0.3  # below threshold
        ),
        # P-modules should be skipped
        ModuleId(0, "attn", "P"): ModuleSignals(
            module_id=ModuleId(0, "attn", "P"), stability_C=0.9, repetition=0.9
        ),
    }

    consolidated = scheduler.consolidate(signals)
    assert mid_qualify in consolidated
    assert mid_no_qualify not in consolidated


def test_consolidation_resets_lora_after_merge():
    """After consolidation, LoRA B matrices are zeroed."""
    torch.manual_seed(42)
    config = ConsolidationConfig(min_stability_C=0.0, min_repetition=0.0, period=100)
    student_cfg = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q"]),
    )
    model = StudentModel(student_cfg)
    p_store = PermanentStore()
    scheduler = ConsolidationScheduler(config, model, p_store)

    # Set LoRA B nonzero
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.5)

    mid = ModuleId(0, "attn", "F")
    signals = {mid: ModuleSignals(module_id=mid, stability_C=0.9, repetition=0.9)}

    scheduler.consolidate(signals)

    # B should be zero after merge
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert torch.allclose(m.lora_B, torch.zeros_like(m.lora_B))

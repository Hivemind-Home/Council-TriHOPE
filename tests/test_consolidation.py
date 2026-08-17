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


def _model_with_trained_lora(target_modules=("q", "k", "v", "o")):
    """Build a tiny student, run a few masked-AdamW steps so LoRA and
    optimizer state are both nonzero, and return (model, optimizer)."""
    from hivemind.optim.masked_adamw import MaskedAdamW

    torch.manual_seed(7)
    cfg = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=list(target_modules)),
    )
    model = StudentModel(cfg)
    optimizer = MaskedAdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        optimizer.zero_grad()
        tokens = torch.randint(0, 64, (2, 8))
        model(tokens).sum().backward()
        optimizer.step()
    return model, optimizer


def test_merge_is_function_preserving():
    """Theorem 2: logits are identical immediately before and after a merge."""
    model, optimizer = _model_with_trained_lora()
    scheduler = ConsolidationScheduler(
        ConsolidationConfig(min_stability_C=0.0, min_repetition=0.0),
        model, PermanentStore(), optimizer=optimizer,
    )
    model.eval()
    probe = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        before = model(probe).clone()

    mid = ModuleId(0, "attn", "F")
    signals = {mid: ModuleSignals(module_id=mid, stability_C=0.9, repetition=0.9)}
    consolidated = scheduler.consolidate(signals)
    assert consolidated == [mid]

    with torch.no_grad():
        after = model(probe)
    torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


def test_merge_resets_adapter_optimizer_state():
    """Corollary 2: merged adapter's Adam moments and counters are zeroed."""
    from hivemind.student.lora_adapter import iter_lora_adapters

    model, optimizer = _model_with_trained_lora()
    scheduler = ConsolidationScheduler(
        ConsolidationConfig(min_stability_C=0.0, min_repetition=0.0),
        model, PermanentStore(), optimizer=optimizer,
    )

    lora_params = [
        p for _n, ad in iter_lora_adapters(model) for p in ad.lora_params
    ]
    # Sanity: training above actually built nonzero moments.
    assert any(
        optimizer.state.get(p) and optimizer.state[p]["exp_avg"].abs().sum() > 0
        for p in lora_params
    )

    mid = ModuleId(0, "attn", "F")
    signals = {mid: ModuleSignals(module_id=mid, stability_C=0.9, repetition=0.9)}
    scheduler.consolidate(signals)

    for p in lora_params:
        state = optimizer.state.get(p)
        if not state:
            continue
        assert torch.all(state["exp_avg"] == 0)
        assert torch.all(state["exp_avg_sq"] == 0)
        assert float(state["step"]) == 0.0
        assert state["coord_step"].dim() == 0
        assert float(state["coord_step"]) == 0.0


def test_first_update_after_merge_uses_fresh_bias_correction():
    """The next accepted adapter update behaves like a literal first step."""
    from hivemind.student.lora_adapter import iter_lora_adapters

    model, optimizer = _model_with_trained_lora(target_modules=("q",))
    scheduler = ConsolidationScheduler(
        ConsolidationConfig(min_stability_C=0.0, min_repetition=0.0),
        model, PermanentStore(), optimizer=optimizer,
    )
    mid = ModuleId(0, "attn", "F")
    signals = {mid: ModuleSignals(module_id=mid, stability_C=0.9, repetition=0.9)}
    scheduler.consolidate(signals)

    (_, adapter), = list(iter_lora_adapters(model))
    p = adapter.lora_a
    before = p.detach().clone()
    grad = torch.randn_like(p)

    # Reference: a fresh AdamW taking its first step on a copy.
    import torch.nn as nn

    q = nn.Parameter(before.clone())
    ref = torch.optim.AdamW([q], lr=1e-2, weight_decay=0.0, foreach=False)
    q.grad = grad.clone()
    ref.step()

    optimizer.zero_grad()
    p.grad = grad.clone()
    optimizer.step()
    torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-6, atol=1e-8)


def _model_with_lora():
    """A tiny student with LoRA on every projection, B initialized nonzero."""
    torch.manual_seed(0)
    model = StudentModel(StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    ))
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.1)
    return model


def test_sustained_mode_merges_on_ema_despite_noisy_instant_cosine():
    """The fix: a module stable over time (high C̄) consolidates even when
    the instantaneous cosine at the period boundary is low."""
    config = ConsolidationConfig(
        min_stability_C=0.7, min_repetition=0.5, period=100, stability_mode="sustained"
    )
    scheduler = ConsolidationScheduler(config, _model_with_lora(), PermanentStore())

    mid = ModuleId(0, "attn", "F")
    signals = {
        mid: ModuleSignals(
            module_id=mid,
            stability_C=0.1,             # noisy/low at THIS boundary step
            stability_C_sustained=0.9,   # but sustained-stable over the window
            repetition=0.8,
        )
    }
    assert mid in scheduler.consolidate(signals)


def test_instant_mode_rejects_the_same_noisy_boundary_step():
    """Same signals under the original ``instant`` mode do NOT merge — this
    is exactly the P≈0 behaviour the fix addresses."""
    config = ConsolidationConfig(
        min_stability_C=0.7, min_repetition=0.5, period=100, stability_mode="instant"
    )
    scheduler = ConsolidationScheduler(config, _model_with_lora(), PermanentStore())

    mid = ModuleId(0, "attn", "F")
    signals = {
        mid: ModuleSignals(
            module_id=mid, stability_C=0.1, stability_C_sustained=0.9, repetition=0.8
        )
    }
    assert mid not in scheduler.consolidate(signals)


def test_sustained_mode_rejects_low_ema_despite_lucky_high_instant():
    """Sustained mode does not consolidate a module that is only aligned at
    this one step (high instant C, low C̄) — it is not a stricter-instant."""
    config = ConsolidationConfig(
        min_stability_C=0.7, min_repetition=0.5, period=100, stability_mode="sustained"
    )
    scheduler = ConsolidationScheduler(config, _model_with_lora(), PermanentStore())

    mid = ModuleId(0, "attn", "F")
    signals = {
        mid: ModuleSignals(
            module_id=mid, stability_C=0.95, stability_C_sustained=0.2, repetition=0.8
        )
    }
    assert mid not in scheduler.consolidate(signals)


def test_instant_is_the_default_mode():
    """Default config re-validates on the instantaneous cosine (no behaviour
    change for existing configs/checkpoints)."""
    config = ConsolidationConfig(min_stability_C=0.7, min_repetition=0.5, period=100)
    assert config.stability_mode == "instant"
    scheduler = ConsolidationScheduler(config, _model_with_lora(), PermanentStore())

    mid = ModuleId(0, "attn", "F")
    # high instant, unset sustained (0.0) → must still merge under default.
    signals = {mid: ModuleSignals(module_id=mid, stability_C=0.9, repetition=0.8)}
    assert mid in scheduler.consolidate(signals)


def test_invalid_stability_mode_rejected():
    import pytest
    with pytest.raises(ValueError, match="stability_mode"):
        ConsolidationConfig(stability_mode="bogus")

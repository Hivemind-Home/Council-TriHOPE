"""Distill-based consolidation strategy — runs an inner KD loop that
trains base to match student+LoRA then resets LoRA.

We check: (a) it runs without raising, (b) LoRA is actually reset
(``lora_B`` returns to zero which is the standard init), (c) the base
weights change between before and after.
"""

from __future__ import annotations

import torch

from hivemind.controller.config import ConsolidationConfig
from hivemind.controller.consolidation import ConsolidationScheduler
from hivemind.controller.module_index import ModuleId
from hivemind.controller.signals import ModuleSignals
from hivemind.stores.permanent import PermanentStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _build_model():
    cfg = StudentConfig(
        vocab_size=32, dim=16, num_layers=1, heads=2, max_seq_len=8,
        lora=LoRAConfig(rank=2, alpha=4.0,
                        target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    return StudentModel(cfg)


def test_distill_merge_runs_and_resets_lora():
    torch.manual_seed(0)
    model = _build_model()
    # Make LoRA non-zero so there's something to fold
    for lora in model.get_all_lora_modules().values():
        with torch.no_grad():
            lora.lora_B.normal_(std=0.01)

    cfg = ConsolidationConfig(
        period=1,
        min_stability_C=0.0,
        min_repetition=0.0,
        merge_strategy="distill",
        distill_iters=2,
        distill_lr=1e-3,
        distill_tau=2.0,
        distill_replay_size=4,
    )
    scheduler = ConsolidationScheduler(cfg, model, PermanentStore())
    # Feed a replay batch
    tokens = torch.randint(0, 32, (2, 8))
    scheduler.record_batch(tokens)

    # Trigger consolidation for the layer-0 ffn F-module
    mid = ModuleId(layer=0, block_type="ffn", param_type="F")
    sig = ModuleSignals(module_id=mid, repetition=1.0, stability_C=1.0)

    base_before = model.blocks[0].ffn.w_up.base.weight.detach().clone()
    out = scheduler.consolidate({mid: sig})
    base_after = model.blocks[0].ffn.w_up.base.weight

    assert mid in out
    # Base moved (non-zero inner loss → gradient step)
    assert not torch.allclose(base_before, base_after)
    # LoRA reset: B is zero post-reset
    assert torch.all(model.blocks[0].ffn.w_up.lora_B == 0)


def test_direct_merge_still_works():
    torch.manual_seed(0)
    model = _build_model()
    cfg = ConsolidationConfig(
        period=1, min_stability_C=0.0, min_repetition=0.0, merge_strategy="direct"
    )
    scheduler = ConsolidationScheduler(cfg, model, PermanentStore())
    mid = ModuleId(layer=0, block_type="attn", param_type="F")
    sig = ModuleSignals(module_id=mid, repetition=1.0, stability_C=1.0)
    out = scheduler.consolidate({mid: sig})
    assert mid in out


def test_distill_falls_back_when_no_replay():
    model = _build_model()
    cfg = ConsolidationConfig(
        period=1, min_stability_C=0.0, min_repetition=0.0, merge_strategy="distill",
    )
    scheduler = ConsolidationScheduler(cfg, model, PermanentStore())
    mid = ModuleId(layer=0, block_type="ffn", param_type="F")
    sig = ModuleSignals(module_id=mid, repetition=1.0, stability_C=1.0)
    # No record_batch called → falls back to direct merge; must not raise.
    out = scheduler.consolidate({mid: sig})
    assert mid in out

"""Tests for consolidation debug controls + baseline switches (B5)."""

from __future__ import annotations

import pytest
import torch

from hivemind.controller.config import (
    AblationConfig,
    ConsolidationConfig,
    DebugConfig,
    PolicyConfig,
)
from hivemind.controller.consolidation import ConsolidationScheduler
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import ModuleSignals
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.stores.permanent import PermanentStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.lora import LoRALinear
from hivemind.student.model import StudentModel


def _model():
    torch.manual_seed(11)
    model = StudentModel(
        StudentConfig(
            vocab_size=64, dim=32, num_layers=2, heads=4,
            lora=LoRAConfig(rank=4, target_modules=["q", "up"]),
        )
    )
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.1)
    return model


class TestDebugConfig:
    def test_invalid_override_raises(self) -> None:
        with pytest.raises(ValueError, match="policy_override"):
            DebugConfig(policy_override="always_x")

    def test_steps_coerced_to_int(self) -> None:
        d = DebugConfig(force_consolidate_steps=["100", 200])
        assert d.force_consolidate_steps == [100, 200]


class TestPolicyOverride:
    def _sig(self, store_would_be_p: bool = False) -> ModuleSignals:
        mid = ModuleId(0, "attn", "F")
        if store_would_be_p:
            return ModuleSignals(
                module_id=mid, grad_norm=1.0, repetition=0.9,
                stability_C=0.9, stability_V=0.0,
            )
        return ModuleSignals(module_id=mid, grad_norm=1.0, surprise=10.0)

    @pytest.mark.parametrize(
        "override,expected", [("always_p", "P"), ("always_f", "F"), ("always_r", "R")]
    )
    def test_override_wins(self, override: str, expected: str) -> None:
        policy = RFPPolicy(PolicyConfig(), override=override)
        sig = self._sig()
        assert policy.decide({sig.module_id: sig})[0].store == expected

    def test_override_skips_confidence_gate(self) -> None:
        policy = RFPPolicy(
            PolicyConfig(),
            ablation=AblationConfig(use_teacher_confidence=True, confidence_threshold=0.9),
            override="always_p",
        )
        sig = self._sig()
        assert policy.decide({sig.module_id: sig}, teacher_confidence=0.0)[0].store == "P"


class TestForceConsolidate:
    def test_merges_all_f_modules_bypassing_thresholds(self) -> None:
        model = _model()
        optimizer = MaskedAdamW(model.parameters(), lr=1e-3)
        # Build nonzero optimizer state on the adapters.
        model(torch.randint(0, 64, (2, 8))).sum().backward()
        optimizer.step()

        scheduler = ConsolidationScheduler(
            # Thresholds intentionally unreachable — force must bypass them.
            ConsolidationConfig(min_stability_C=99.0, min_repetition=99.0),
            model, PermanentStore(), optimizer=optimizer,
        )
        merged = scheduler.force_consolidate()
        assert set(merged) == {
            ModuleId(0, "attn", "F"), ModuleId(0, "ffn", "F"),
            ModuleId(1, "attn", "F"), ModuleId(1, "ffn", "F"),
        }
        # LoRA reset + optimizer state zeroed.
        for m in model.modules():
            if isinstance(m, LoRALinear):
                assert torch.all(m.lora_B == 0)
                state = optimizer.state.get(m.lora_A)
                if state:
                    assert torch.all(state["exp_avg"] == 0)

    def test_specific_modules_only(self) -> None:
        model = _model()
        scheduler = ConsolidationScheduler(
            ConsolidationConfig(), model, PermanentStore()
        )
        target = ModuleId(1, "ffn", "F")
        merged = scheduler.force_consolidate([target, ModuleId(0, "attn", "P")])
        assert merged == [target]

    def test_all_f_modules_lists_lora_blocks(self) -> None:
        model = _model()
        scheduler = ConsolidationScheduler(ConsolidationConfig(), model, PermanentStore())
        assert len(scheduler.all_f_modules()) == 4  # 2 layers × (attn, ffn)


class TestTaggedCheckpoints:
    def test_tagged_dirs_separate_from_latest_and_prune(self, tmp_path) -> None:
        from hivemind.checkpoint import CheckpointConfig, CheckpointManager
        from hivemind.controller.config import ControllerConfig
        from hivemind.controller.module_index import build_module_index
        from hivemind.controller.signals import SignalComputer
        from hivemind.stores.retrieval import RetrievalStore

        model = _model()
        optimizer = MaskedAdamW(model.parameters(), lr=1e-3)
        modules = build_module_index(model)
        sc = SignalComputer(ControllerConfig(), modules)
        r_store = RetrievalStore(max_size=10)

        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path), keep_last=2, keep_tagged=2)
        )
        common = dict(
            student=model, optimizer=optimizer, signal_computer=sc, r_store=r_store
        )
        for step in (10, 20, 30):
            mgr.save(step=step, **common)
        for step in (15, 25, 35):
            mgr.save(step=step, tag="pre_merge", **common)

        names = sorted(
            p.name
            for p in tmp_path.iterdir()
            if p.is_dir() and not p.is_symlink()
        )
        # keep_last=2 untagged, keep_tagged=2 tagged.
        assert names == [
            "step_00000020",
            "step_00000025_pre_merge",
            "step_00000030",
            "step_00000035_pre_merge",
        ]
        # latest resolves to the newest UNTAGGED dir.
        assert mgr._latest().name == "step_00000030"
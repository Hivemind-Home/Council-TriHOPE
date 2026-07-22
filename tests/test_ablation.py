"""Tests for per-signal ablation switches and the teacher-confidence gate."""

from __future__ import annotations

import pytest

from hivemind.controller.config import (
    AblationConfig,
    ControllerConfig,
    PolicyConfig,
)
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import ModuleSignals


def _sig(
    mid: ModuleId,
    surprise: float = 0.0,
    repetition: float = 0.0,
    C: float = 0.0,
    V: float = 1.0,
    grad_norm: float = 1.0,
) -> ModuleSignals:
    return ModuleSignals(
        module_id=mid,
        grad_norm=grad_norm,
        surprise=surprise,
        repetition=repetition,
        stability_C=C,
        stability_V=V,
    )


MID = ModuleId(0, "attn", "F")


class TestAblationConfigValidation:
    def test_unknown_signal_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown signal"):
            AblationConfig(disable_signals=["surprize"])

    def test_aliases_canonicalize(self) -> None:
        ab = AblationConfig(disable_signals=["cosine", "volatility"])
        assert ab.disable_signals == ["stability_C", "stability_V"]

    def test_unknown_store_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown store"):
            AblationConfig(disable_stores=["F"])

    def test_teacher_confidence_disable_forces_off(self) -> None:
        ab = AblationConfig(
            disable_signals=["teacher_confidence"], use_teacher_confidence=True
        )
        assert ab.use_teacher_confidence is False


class TestDefaultBehaviorUnchanged:
    def test_default_ablation_identical_decisions(self) -> None:
        base = RFPPolicy(PolicyConfig())
        with_ab = RFPPolicy(PolicyConfig(), ablation=AblationConfig())
        cases = [
            _sig(MID, surprise=5.0, repetition=0.1),          # R
            _sig(MID, repetition=0.6, C=0.7, V=0.1),          # P
            _sig(MID, surprise=1.0, repetition=0.4),          # F
        ]
        for sig in cases:
            a = base.decide({MID: sig})[0].store
            b = with_ab.decide({MID: sig})[0].store
            assert a == b


class TestDisableStores:
    def test_no_retrieval_falls_to_f(self) -> None:
        policy = RFPPolicy(
            PolicyConfig(), ablation=AblationConfig(disable_stores=["R"])
        )
        sig = _sig(MID, surprise=5.0, repetition=0.1)  # would be R
        assert policy.decide({MID: sig})[0].store == "F"

    def test_no_consolidation_falls_to_f(self) -> None:
        policy = RFPPolicy(
            PolicyConfig(), ablation=AblationConfig(disable_stores=["P"])
        )
        sig = _sig(MID, repetition=0.6, C=0.7, V=0.1)  # would be P
        assert policy.decide({MID: sig})[0].store == "F"


class TestTopMOverride:
    def test_override_limits_selection(self) -> None:
        policy = RFPPolicy(
            PolicyConfig(top_m_modules=6), ablation=AblationConfig(top_m_override=1)
        )
        signals = {
            ModuleId(i, "attn", "F"): _sig(
                ModuleId(i, "attn", "F"), grad_norm=float(i + 1)
            )
            for i in range(4)
        }
        actions = policy.decide(signals)
        assert len(actions) == 1
        assert actions[0].module_id == ModuleId(3, "attn", "F")  # largest norm


class TestConfidenceGate:
    def test_no_p_demotes_p_to_f(self) -> None:
        policy = RFPPolicy(
            PolicyConfig(),
            ablation=AblationConfig(
                use_teacher_confidence=True, confidence_threshold=0.5
            ),
        )
        sig = _sig(MID, repetition=0.6, C=0.7, V=0.1)  # qualifies for P
        assert policy.decide({MID: sig}, teacher_confidence=0.9)[0].store == "P"
        assert policy.decide({MID: sig}, teacher_confidence=0.2)[0].store == "F"

    def test_force_r_routes_everything_to_r(self) -> None:
        policy = RFPPolicy(
            PolicyConfig(),
            ablation=AblationConfig(
                use_teacher_confidence=True,
                confidence_threshold=0.5,
                confidence_gate="force_r",
            ),
        )
        sig = _sig(MID, repetition=0.6, C=0.7, V=0.1)
        assert policy.decide({MID: sig}, teacher_confidence=0.1)[0].store == "R"

    def test_gate_off_by_default(self) -> None:
        policy = RFPPolicy(PolicyConfig(), ablation=AblationConfig())
        sig = _sig(MID, repetition=0.6, C=0.7, V=0.1)
        assert policy.decide({MID: sig}, teacher_confidence=0.0)[0].store == "P"


class TestNeutralValues:
    """Disabled signals pin to neutrals that degrade the policy sensibly."""

    def _computer(self, disable: list[str]):
        import torch

        from hivemind.controller.module_index import build_module_index
        from hivemind.controller.signals import SignalComputer
        from hivemind.optim.masked_adamw import MaskedAdamW
        from hivemind.student.config import LoRAConfig, StudentConfig
        from hivemind.student.model import StudentModel

        torch.manual_seed(0)
        model = StudentModel(
            StudentConfig(
                vocab_size=64, dim=32, num_layers=1, heads=4,
                lora=LoRAConfig(rank=4, target_modules=["q"]),
            )
        )
        cfg = ControllerConfig(ablation=AblationConfig(disable_signals=disable))
        modules = build_module_index(model)
        computer = SignalComputer(cfg, modules)
        opt = MaskedAdamW(model.parameters(), lr=1e-3)
        # Two plain steps to build optimizer state.
        for _ in range(2):
            opt.zero_grad()
            tokens = torch.randint(0, 64, (2, 8))
            model(tokens).sum().backward()
            opt.step()
        opt.zero_grad()
        tokens = torch.randint(0, 64, (2, 8))
        model(tokens).sum().backward()
        signals = computer.compute_all(opt, modules, bucket_id=1)
        return signals, cfg

    def test_disabled_surprise_pinned_to_threshold(self) -> None:
        signals, cfg = self._computer(["surprise"])
        for sig in signals.values():
            if sig.grad_norm > 0:
                assert sig.surprise == cfg.policy.surprise_high

    def test_disabled_repetition_pinned_to_medium(self) -> None:
        signals, cfg = self._computer(["repetition"])
        for sig in signals.values():
            if sig.grad_norm > 0:
                assert sig.repetition == cfg.policy.repetition_medium
                assert sig.repetition_components == {}

    def test_disabled_cosine_pinned_high(self) -> None:
        signals, cfg = self._computer(["cosine"])
        for sig in signals.values():
            if sig.grad_norm > 0:
                assert sig.stability_C == cfg.policy.stability_high_C

    def test_disabled_volatility_pinned_low(self) -> None:
        signals, cfg = self._computer(["volatility"])
        for sig in signals.values():
            if sig.grad_norm > 0:
                assert sig.stability_V == cfg.policy.stability_low_V


class TestComponentRenormalization:
    def test_disabled_component_zeroed_and_renormalized(self) -> None:

        from hivemind.controller.config import RepetitionConfig
        from hivemind.controller.module_index import build_module_index
        from hivemind.controller.signals import SignalComputer
        from hivemind.student.config import LoRAConfig, StudentConfig
        from hivemind.student.model import StudentModel

        model = StudentModel(
            StudentConfig(
                vocab_size=64, dim=32, num_layers=1, heads=4,
                lora=LoRAConfig(rank=4, target_modules=["q"]),
            )
        )
        cfg = ControllerConfig(
            repetition=RepetitionConfig(lambda_mom=0.3, lambda_hash=0.5, lambda_ret=0.2),
            ablation=AblationConfig(disable_signals=["repetition_mom"]),
        )
        computer = SignalComputer(cfg, build_module_index(model))
        assert computer._rep_cfg.lambda_mom == 0.0
        assert computer._rep_cfg.lambda_hash == 0.5  # renormalized inside FusedRepetition

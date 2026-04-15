"""Tests for R/F/P routing policy."""

from hivemind.controller.config import PolicyConfig
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import ModuleSignals


def _make_signals(
    layer: int = 0,
    block: str = "attn",
    ptype: str = "F",
    grad_norm: float = 1.0,
    surprise: float = 1.0,
    repetition: float = 0.5,
    stability_C: float = 0.5,
    stability_V: float = 0.3,
) -> tuple[ModuleId, ModuleSignals]:
    mid = ModuleId(layer, block, ptype)
    return mid, ModuleSignals(
        module_id=mid,
        grad_norm=grad_norm,
        surprise=surprise,
        repetition=repetition,
        stability_C=stability_C,
        stability_V=stability_V,
    )


def test_r_route_high_surprise_low_repetition():
    """High surprise + low repetition → R (retrieval)."""
    policy = RFPPolicy(PolicyConfig(
        top_m_modules=4,
        surprise_high=2.0,
        repetition_low=0.3,
        repetition_medium=0.5,
        stability_high_C=0.5,
        stability_low_V=0.3,
    ))
    mid, sig = _make_signals(surprise=5.0, repetition=0.1)
    signals = {mid: sig}

    actions = policy.decide(signals)
    assert len(actions) == 1
    assert actions[0].store == "R"


def test_f_route_moderate_signals():
    """Moderate signals → F (fast/LoRA)."""
    policy = RFPPolicy(PolicyConfig(
        top_m_modules=4,
        surprise_high=2.0,
        repetition_low=0.3,
        repetition_medium=0.5,
        stability_high_C=0.5,
        stability_low_V=0.3,
    ))
    # Medium repetition, not stable enough for P
    mid, sig = _make_signals(surprise=1.5, repetition=0.6, stability_C=0.3, stability_V=0.5)
    signals = {mid: sig}

    actions = policy.decide(signals)
    assert len(actions) == 1
    assert actions[0].store == "F"


def test_p_route_high_repetition_high_stability():
    """High repetition + high stability → P (permanent)."""
    policy = RFPPolicy(PolicyConfig(
        top_m_modules=4,
        surprise_high=2.0,
        repetition_low=0.3,
        repetition_medium=0.5,
        stability_high_C=0.5,
        stability_low_V=0.3,
    ))
    mid, sig = _make_signals(surprise=1.0, repetition=0.8, stability_C=0.8, stability_V=0.1)
    signals = {mid: sig}

    actions = policy.decide(signals)
    assert len(actions) == 1
    assert actions[0].store == "P"


def test_top_m_selection():
    """Only Top-M modules by gradient norm get actions."""
    policy = RFPPolicy(PolicyConfig(top_m_modules=2))

    signals = {}
    for i in range(5):
        mid, sig = _make_signals(layer=i, grad_norm=float(i + 1))
        signals[mid] = sig

    actions = policy.decide(signals)
    # Should pick top 2 by grad_norm (layers 4 and 3)
    assert len(actions) == 2
    action_layers = {a.module_id.layer for a in actions}
    assert 4 in action_layers
    assert 3 in action_layers


def test_zero_grad_modules_skipped():
    """Modules with zero gradient are skipped."""
    policy = RFPPolicy(PolicyConfig(top_m_modules=4))
    mid, sig = _make_signals(grad_norm=0.0)
    signals = {mid: sig}

    actions = policy.decide(signals)
    assert len(actions) == 0

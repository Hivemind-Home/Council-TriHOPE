"""Signal moments (controller/moments.py): the controller's own continuous
Adam-style m/v, alive on coordinates the write mask keeps closed.

Regression for the first real-stream run, where every base-weight module
read m = v = 0 forever (surprise pinned at s_max, C̄ = 0, P never fired).
"""

from __future__ import annotations

import torch

from hivemind.controller.config import ControllerConfig, MomentsConfig
from hivemind.controller.module_index import build_module_index
from hivemind.controller.moments import SignalMomentTracker
from hivemind.controller.signals import SignalComputer
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _stack(source: str, stride: int = 1):
    torch.manual_seed(0)
    cfg = StudentConfig(
        vocab_size=64, dim=32, num_layers=2, heads=4, max_seq_len=16,
        lora=LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "v", "up", "down"]),
    )
    model = StudentModel(cfg)
    modules = build_module_index(model)
    opt = MaskedAdamW(model.parameters(), lr=1e-3)
    for m in modules:
        opt.register_controller_params(m.params)
    ctrl = ControllerConfig(moments=MomentsConfig(source=source, sketch_stride=stride))
    sig = SignalComputer(ctrl, modules)
    return model, modules, opt, sig


def _step(model, opt, sig, modules, *, open_all: bool, seed: int):
    torch.manual_seed(seed)
    tokens = torch.randint(0, 64, (2, 16))
    opt.zero_grad()
    model(tokens).float().pow(2).mean().backward()
    out = sig.compute_all(opt, modules, bucket_id=1)
    # The loop stages a mask for every registered param each step; ``False``
    # is what a closed module gets (Theorem 1: its state is never touched).
    opt.set_masks({p: open_all for m in modules for p in m.params})
    opt.step()
    opt.clear_masks()
    return out


def test_tracked_equals_adam_on_open_coordinates():
    """Same recursion, same inputs: on always-open coordinates the tracked
    (uncorrected) moments ARE Adam's exp_avg / exp_avg_sq."""
    model, modules, opt, sig = _stack("tracked")
    for i in range(4):
        _step(model, opt, sig, modules, open_all=True, seed=i)
    tr = sig.moments
    checked = 0
    for m in modules:
        for p in m.params:
            name = sig._param_names[id(p)]  # unique tracker key (LoRA A/B share a name)
            state = opt.state[p]
            assert torch.allclose(tr._m[name], state["exp_avg"].flatten().float(), atol=1e-7)
            assert torch.allclose(tr._v[name], state["exp_avg_sq"].flatten().float(), atol=1e-9)
            checked += 1
    assert checked > 0 and tr.step == 4


def test_closed_modules_are_dead_with_optimizer_moments_and_alive_when_tracked():
    """Nothing is ever opened (the P-deadlock situation). With the optimizer's
    own moments every module reads S = s_max and C = 0 forever; with tracked
    moments surprise settles and the cosine is live after two steps."""
    torch.manual_seed(0)
    dead_S, dead_C, live_S, live_C = [], [], [], []
    for source, S_out, C_out in (("optimizer", dead_S, dead_C), ("tracked", live_S, live_C)):
        model, modules, opt, sig = _stack(source)
        for i in range(6):
            out = _step(model, opt, sig, modules, open_all=False, seed=0)  # same batch each step
        for s in out.values():
            if s.grad_norm > 0:
                S_out.append(s.surprise)
                C_out.append(s.stability_C)
    # Optimizer moments of a never-opened module stay at zero: the cosine is
    # a warmup zero forever, and surprise is g²/ε — pinned at s_max for any
    # gradient of realistic scale (the real run), tiny for tiny gradients.
    # Either way it carries no history. Tracked moments see the same
    # gradient six times: the cosine is ~1 and surprise is ~g²/g² ≈ 1.
    assert dead_C and all(C == 0.0 for C in dead_C)
    # (Not exactly 1: embeddings / norms / lm_head are unregistered, stay open
    # and keep moving, so the routed modules' gradient drifts a little.)
    assert live_C and all(C > 0.5 for C in live_C), live_C[:4]
    s_max = ControllerConfig().surprise.s_max
    assert all(S < s_max for S in live_S), live_S[:4]
    assert max(live_S) < 2.0, live_S[:4]


def test_sketch_stride_tracks_the_full_signal():
    stride1 = _stack("tracked", stride=1)
    stride4 = _stack("tracked", stride=4)
    outs = []
    for model, modules, opt, sig in (stride1, stride4):
        torch.manual_seed(0)
        for i in range(5):
            out = _step(model, opt, sig, modules, open_all=False, seed=i)
        outs.append(out)
    sizes = {m.id: m.num_params for m in stride1[1]}
    for mid in outs[0]:
        a, b = outs[0][mid], outs[1][mid]
        if a.grad_norm == 0 or sizes[mid] < 2048:
            continue  # a 32-coordinate sketch of a rank-4 adapter is too small to compare
        assert abs(a.stability_C - b.stability_C) < 0.15, (mid, a.stability_C, b.stability_C)
        assert a.grad_norm == b.grad_norm  # Top-M ranking uses the full gradient


def test_moments_state_dict_roundtrip():
    model, modules, opt, sig = _stack("tracked", stride=2)
    for i in range(3):
        _step(model, opt, sig, modules, open_all=False, seed=i)
    state = sig.state_dict()
    assert state["moments"]["step"] == 3
    model2, modules2, opt2, sig2 = _stack("tracked", stride=2)
    sig2.load_state_dict(state)
    assert sig2.moments.step == 3
    for name in sig.moments.names():
        assert torch.equal(sig.moments._m[name], sig2.moments._m[name])


def test_tracker_bias_correction():
    tr = SignalMomentTracker(betas=(0.9, 0.999), sketch_stride=1)
    g = torch.ones(8)
    m, v = tr.read("w", g)
    assert float(m.abs().sum()) == 0.0 and float(v.abs().sum()) == 0.0
    tr.update("w", g)
    tr.advance()
    m, v = tr.read("w", g)
    # After one step of a constant gradient the corrected estimate is the gradient itself.
    assert torch.allclose(m, g) and torch.allclose(v, g * g)


def test_moments_config_validation():
    import pytest

    with pytest.raises(ValueError):
        MomentsConfig(source="shadow")
    with pytest.raises(ValueError):
        MomentsConfig(sketch_stride=0)

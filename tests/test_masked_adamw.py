"""Tests for MaskedAdamW exact state isolation (paper Theorem 1 / Corollaries).

Covers:
- Lemma 1: all-open reduction to ordinary torch.optim.AdamW.
- Theorem 1: closed coordinates are bit-identical across params, moments,
  AMSGrad max, coord_step — including decoupled weight decay.
- Corollary 1: coordinate-local bias correction after reopening.
- Corollary 2: reset_state_for_params zeroes everything.
- Broadcast masks, default-closed registration, state_dict round-trip.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from hivemind.optim.masked_adamw import MaskedAdamW

LR = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8
WD = 0.1  # large so any weight-decay leak onto closed coords is visible


def _make_params(seed: int = 0, shapes=((4, 6), (5,))) -> list[nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(*s, generator=g)) for s in shapes]


def _set_grads(params: list[nn.Parameter], seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g).to(p.dtype)


class TestAllOpenReduction:
    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_matches_torch_adamw(self, amsgrad: bool) -> None:
        ours_params = _make_params(seed=1)
        ref_params = _make_params(seed=1)
        ours = MaskedAdamW(
            ours_params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, amsgrad=amsgrad
        )
        ref = torch.optim.AdamW(
            ref_params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, amsgrad=amsgrad, foreach=False
        )
        for step in range(12):
            _set_grads(ours_params, seed=100 + step)
            _set_grads(ref_params, seed=100 + step)
            ours.step()
            ref.step()
        for po, pr in zip(ours_params, ref_params):
            torch.testing.assert_close(po, pr, rtol=1e-6, atol=1e-7)
            so, sr = ours.state[po], ref.state[pr]
            torch.testing.assert_close(so["exp_avg"], sr["exp_avg"], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(so["exp_avg_sq"], sr["exp_avg_sq"], rtol=1e-6, atol=1e-7)
            if amsgrad:
                torch.testing.assert_close(
                    so["max_exp_avg_sq"], sr["max_exp_avg_sq"], rtol=1e-6, atol=1e-7
                )
            assert float(so["coord_step"]) == 12.0

    def test_explicit_true_mask_same_as_no_mask(self) -> None:
        a_params = _make_params(seed=2)
        b_params = _make_params(seed=2)
        a = MaskedAdamW(a_params, lr=LR, weight_decay=WD)
        b = MaskedAdamW(b_params, lr=LR, weight_decay=WD)
        for step in range(5):
            _set_grads(a_params, seed=step)
            _set_grads(b_params, seed=step)
            b.set_masks({p: MaskedAdamW.FULLY_OPEN for p in b_params})
            a.step()
            b.step()
        for pa, pb in zip(a_params, b_params):
            assert torch.equal(pa, pb)


class TestClosedCoordinateIsolation:
    def test_closed_coords_bit_identical(self) -> None:
        (p,) = _make_params(seed=3, shapes=((6, 8),))
        opt = MaskedAdamW([p], lr=LR, weight_decay=WD, amsgrad=True)
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[:3] = True  # open first 3 rows only
        closed = ~mask

        # One all-open step to build nonzero state everywhere first.
        _set_grads([p], seed=0)
        opt.step()

        snap_p = p.detach().clone()
        snap_m = opt.state[p]["exp_avg"].clone()
        snap_v = opt.state[p]["exp_avg_sq"].clone()
        snap_max = opt.state[p]["max_exp_avg_sq"].clone()

        for step in range(6):
            _set_grads([p], seed=10 + step)
            opt.set_masks({p: mask})
            opt.step()

        assert torch.equal(p.detach()[closed], snap_p[closed])
        assert torch.equal(opt.state[p]["exp_avg"][closed], snap_m[closed])
        assert torch.equal(opt.state[p]["exp_avg_sq"][closed], snap_v[closed])
        assert torch.equal(opt.state[p]["max_exp_avg_sq"][closed], snap_max[closed])
        cs = opt.state[p]["coord_step"]
        assert torch.all(cs[closed] == 1.0)  # only the initial all-open step
        assert torch.all(cs[~closed] == 7.0)
        # Open rows must have moved (sanity that the step did something).
        assert not torch.equal(p.detach()[mask], snap_p[mask])

    def test_fully_closed_param_never_initializes_state(self) -> None:
        (p,) = _make_params(seed=4, shapes=((3, 3),))
        opt = MaskedAdamW([p], lr=LR, weight_decay=WD)
        snap = p.detach().clone()
        for step in range(3):
            _set_grads([p], seed=step)
            opt.set_masks({p: MaskedAdamW.FULLY_CLOSED})
            opt.step()
        assert torch.equal(p.detach(), snap)
        assert len(opt.state) == 0  # never lazily initialized

    def test_registered_params_default_closed(self) -> None:
        p_ctrl, p_free = _make_params(seed=5, shapes=((4, 4), (4, 4)))
        opt = MaskedAdamW([p_ctrl, p_free], lr=LR, weight_decay=WD)
        opt.register_controller_params([p_ctrl])
        snap_ctrl = p_ctrl.detach().clone()
        snap_free = p_free.detach().clone()
        _set_grads([p_ctrl, p_free], seed=0)
        opt.step()  # no masks staged
        assert torch.equal(p_ctrl.detach(), snap_ctrl)
        assert not torch.equal(p_free.detach(), snap_free)

    def test_masks_cleared_after_step(self) -> None:
        (p,) = _make_params(seed=6, shapes=((4,),))
        opt = MaskedAdamW([p], lr=LR, weight_decay=0.0)
        opt.register_controller_params([p])
        _set_grads([p], seed=0)
        opt.set_masks({p: MaskedAdamW.FULLY_OPEN})
        opt.step()
        moved = p.detach().clone()
        _set_grads([p], seed=1)
        opt.step()  # mask consumed last step -> default closed again
        assert torch.equal(p.detach(), moved)


class TestCoordinateLocalBiasCorrection:
    def test_reopened_coord_uses_own_count(self) -> None:
        """Corollary 1: after r accepted updates, the next uses 1 - beta^(r+1)."""
        p = nn.Parameter(torch.zeros(2))
        opt = MaskedAdamW([p], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0)
        opt.register_controller_params([p])

        grads = [torch.tensor([0.5, -0.3]), torch.tensor([1.0, 0.2]), torch.tensor([-0.7, 0.9])]
        open_coord0 = torch.tensor([True, False])
        open_coord1 = torch.tensor([False, True])
        # coord0 open at calls 0,1; closed at call 2. coord1 open only at call 2.
        schedule = [open_coord0, open_coord0, open_coord1]

        # Hand-rolled per-coordinate AdamW reference.
        theta = torch.zeros(2)
        m = torch.zeros(2)
        v = torch.zeros(2)
        n = torch.zeros(2)
        for grad, mask in zip(grads, schedule):
            p.grad = grad.clone()
            opt.set_masks({p: mask})
            opt.step()
            for i in range(2):
                if not mask[i]:
                    continue
                n[i] += 1
                m[i] = BETAS[0] * m[i] + (1 - BETAS[0]) * grad[i]
                v[i] = BETAS[1] * v[i] + (1 - BETAS[1]) * grad[i] ** 2
                bc1 = 1 - BETAS[0] ** n[i].item()
                bc2 = 1 - BETAS[1] ** n[i].item()
                theta[i] = theta[i] - LR / bc1 * m[i] / (math.sqrt(v[i] / bc2) + EPS)

        torch.testing.assert_close(p.detach(), theta, rtol=1e-5, atol=1e-7)
        assert torch.equal(opt.state[p]["coord_step"], n)
        # coord1's single update used first-update bias correction regardless of
        # the two optimizer calls that happened while it was closed.
        assert float(opt.state[p]["coord_step"][1]) == 1.0


class TestBroadcastMasks:
    def test_row_mask_equals_expanded_mask(self) -> None:
        a_params = _make_params(seed=7, shapes=((4, 6),))
        b_params = _make_params(seed=7, shapes=((4, 6),))
        a = MaskedAdamW(a_params, lr=LR, weight_decay=WD)
        b = MaskedAdamW(b_params, lr=LR, weight_decay=WD)
        row_mask = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(1)  # [4, 1]
        full_mask = row_mask.expand(4, 6).contiguous()
        for step in range(4):
            _set_grads(a_params, seed=step)
            _set_grads(b_params, seed=step)
            a.set_masks({a_params[0]: row_mask})
            b.set_masks({b_params[0]: full_mask})
            a.step()
            b.step()
        assert torch.equal(a_params[0].detach(), b_params[0].detach())
        assert torch.equal(
            a.state[a_params[0]]["coord_step"], b.state[b_params[0]]["coord_step"]
        )


class TestResetState:
    def test_reset_zeroes_everything(self) -> None:
        (p,) = _make_params(seed=8, shapes=((4, 4),))
        opt = MaskedAdamW([p], lr=LR, weight_decay=WD, amsgrad=True)
        for step in range(3):
            _set_grads([p], seed=step)
            mask = torch.rand_like(p) > 0.5
            opt.set_masks({p: mask})
            opt.step()
        opt.reset_state_for_params([p])
        state = opt.state[p]
        assert torch.all(state["exp_avg"] == 0)
        assert torch.all(state["exp_avg_sq"] == 0)
        assert torch.all(state["max_exp_avg_sq"] == 0)
        assert state["coord_step"].dim() == 0 and float(state["coord_step"]) == 0.0
        assert float(state["step"]) == 0.0

    def test_next_update_after_reset_is_first_update(self) -> None:
        (p,) = _make_params(seed=9, shapes=((4,),))
        opt = MaskedAdamW([p], lr=LR, weight_decay=0.0)
        for step in range(5):
            _set_grads([p], seed=step)
            opt.step()
        opt.reset_state_for_params([p])
        p_before = p.detach().clone()
        p.grad = torch.full_like(p, 0.25)
        opt.step()
        # Fresh reference optimizer taking its literal first step.
        q = nn.Parameter(p_before.clone())
        ref = torch.optim.AdamW([q], lr=LR, weight_decay=0.0, foreach=False)
        q.grad = torch.full_like(q, 0.25)
        ref.step()
        torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-6, atol=1e-8)


class TestStateDictRoundTrip:
    def test_roundtrip_preserves_coord_step_and_trajectory(self) -> None:
        params_a = _make_params(seed=10, shapes=((4, 6), (3,)))
        opt_a = MaskedAdamW(params_a, lr=LR, weight_decay=WD)
        partial = torch.zeros(4, 6, dtype=torch.bool)
        partial[:2] = True
        for step in range(3):
            _set_grads(params_a, seed=step)
            opt_a.set_masks({params_a[0]: partial})  # second param stays uniform
            opt_a.step()

        # Clone into a fresh model/optimizer pair and restore. deepcopy mirrors
        # a save/load through torch.save (in-process state_dicts hold live refs).
        import copy

        params_b = [nn.Parameter(p.detach().clone()) for p in params_a]
        opt_b = MaskedAdamW(params_b, lr=LR, weight_decay=WD)
        opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))

        cs_a = opt_a.state[params_a[0]]["coord_step"]
        cs_b = opt_b.state[params_b[0]]["coord_step"]
        assert cs_b.dtype == torch.float32
        assert torch.equal(cs_a, cs_b)
        # Uniform param's scalar counter also survives.
        assert opt_b.state[params_b[1]]["coord_step"].dim() == 0

        # Continue both for 3 more steps: identical trajectories.
        for step in range(3, 6):
            _set_grads(params_a, seed=step)
            for pa, pb in zip(params_a, params_b):
                pb.grad = pa.grad.clone()
            opt_a.set_masks({params_a[0]: partial})
            opt_b.set_masks({params_b[0]: partial})
            opt_a.step()
            opt_b.step()
        for pa, pb in zip(params_a, params_b):
            assert torch.equal(pa.detach(), pb.detach())

    def test_bf16_param_coord_step_not_corrupted(self) -> None:
        p = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        opt = MaskedAdamW([p], lr=LR, weight_decay=0.0)
        partial = torch.zeros(4, 4, dtype=torch.bool)
        partial[0] = True
        # Drive coord_step above 256 (bf16 integer precision limit).
        for step in range(300):
            p.grad = torch.randn_like(p)
            opt.set_masks({p: partial})
            opt.step()
        q = nn.Parameter(p.detach().clone())
        opt2 = MaskedAdamW([q], lr=LR, weight_decay=0.0)
        opt2.load_state_dict(opt.state_dict())
        cs = opt2.state[q]["coord_step"]
        assert cs.dtype == torch.float32
        assert float(cs[0, 0]) == 300.0


class TestConstructorRejections:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fused": True},
            {"foreach": True},
            {"capturable": True},
            {"differentiable": True},
            {"maximize": True},
        ],
    )
    def test_rejects_unsupported_modes(self, kwargs: dict) -> None:
        params = _make_params()
        with pytest.raises(ValueError):
            MaskedAdamW(params, lr=LR, **kwargs)

    def test_accepts_explicit_false_modes(self) -> None:
        params = _make_params()
        MaskedAdamW(params, lr=LR, fused=False, maximize=False)

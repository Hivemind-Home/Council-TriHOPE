"""Gradient-routing baseline (task T6): teacher-labelled rank slices, the
slice masks, the masked optimizer-state reset, and the slice ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from hivemind.controller.config import DebugConfig, PolicyConfig
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import ModuleSignals
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.stores.fast import FastStore
from hivemind.student.lora import LoRALinear
from hivemind.training import run_training_loop


class TestSliceMasks:
    def test_slices_are_disjoint_and_cover_the_used_rank(self) -> None:
        assert [FastStore.slice_bounds(7, i, 3) for i in range(3)] == [(0, 2), (2, 4), (4, 6)]
        assert FastStore.slice_bounds(16, 4, 5) == (12, 15)  # one component unused
        with pytest.raises(ValueError, match="lora.rank"):
            FastStore.slice_bounds(3, 0, 4)

    def test_masks_open_only_the_slice(self) -> None:
        lin = LoRALinear(8, 6, rank=4, alpha=8.0)
        from hivemind.student.lora_adapter import get_adapter

        ad = get_adapter(lin)
        store = FastStore()
        mask_a, mask_b = store.compute_slice_masks(ad, 1, 2)
        assert mask_a.shape == ad.lora_a.shape and mask_b.shape == ad.lora_b.shape
        assert mask_a[:, 0].tolist() == [False, False, True, True]
        assert mask_b[0].tolist() == [False, False, True, True]
        other_a, other_b = store.compute_slice_masks(ad, 0, 2)
        assert not (mask_a & other_a).any() and not (mask_b & other_b).any()
        assert (mask_a | other_a).all()


class TestPolicyMode:
    def test_all_f_modules_get_sliced_f_actions(self) -> None:
        pol = RFPPolicy(PolicyConfig(mode="teacher_partition", top_m_modules=1))
        sigs = {}
        for layer in (0, 1):
            for block in ("attn", "ffn"):
                for ptype in ("F", "P"):
                    mid = ModuleId(layer, block, ptype)
                    sigs[mid] = ModuleSignals(module_id=mid, grad_norm=1.0, surprise=9.0)
        acts = pol.decide(sigs, teacher_index=2, num_teachers=5)
        assert len(acts) == 4 and all(a.store == "F" for a in acts)
        assert all(a.teacher_slot == (2, 5) for a in acts)
        assert all(a.module_id.param_type == "F" for a in acts)
        with pytest.raises(ValueError, match="teacher_index"):
            pol.decide(sigs)

    def test_config_validation(self) -> None:
        DebugConfig(ablate_teacher_slice={"step": 3, "teacher": "x"})
        with pytest.raises(ValueError, match="ablate_teacher_slice"):
            DebugConfig(ablate_teacher_slice={"step": 3})


class TestMaskedReset:
    def test_reset_touches_only_masked_coords_and_not_step(self) -> None:
        p = torch.nn.Parameter(torch.randn(4, 3))
        opt = MaskedAdamW([p], lr=1e-2)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
        state = opt.state[p]
        before = {k: v.clone() for k, v in state.items()}
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[1:3] = True
        opt.reset_state_for_coords(p, mask)
        for key in ("exp_avg", "exp_avg_sq"):
            assert (state[key][mask] == 0).all()
            assert torch.equal(state[key][~mask], before[key][~mask])
        cs = state["coord_step"]
        assert cs.shape == p.shape and (cs[mask] == 0).all() and (cs[~mask] == 3).all()
        assert torch.equal(state["step"], before["step"])


def _cfg(tmp_path: Path, ablate_step: int | None) -> OmegaConf:
    debug = {}
    if ablate_step is not None:
        debug["ablate_teacher_slice"] = {"step": ablate_step, "teacher": "teacher_1"}
    return OmegaConf.create(
        {
            "model": {"vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                      "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0}},
            "teachers": {"num_teachers": 2, "mode": "synthetic"},
            "controller": {"policy": {"mode": "teacher_partition"},
                           "consolidation": {"period": 0}, "debug": debug},
            "data": {"source": "synthetic", "dataset_size": 20, "seq_len": 16,
                     "vocab_size": 64, "batch_size": 4},
            "train": {"steps": 8, "device": "cpu", "seed": 3, "log_interval": 1},
            "optim": {"lr": 1e-2},
            "checkpoint": {"enabled": True, "dir": str(tmp_path / "ckpt"), "save_every": 100},
            "logging": {"enabled": True, "backend": "json",
                        "path": str(tmp_path / "metrics.jsonl"),
                        "events_path": str(tmp_path / "events.jsonl")},
        }
    )


def test_end_to_end_partition_and_ablation(tmp_path: Path) -> None:
    run_training_loop(_cfg(tmp_path, ablate_step=5), device=torch.device("cpu"))
    events = [json.loads(ln) for ln in (tmp_path / "events.jsonl").read_text().splitlines()]
    decisions = [e for e in events if e["type"] == "decision"]
    assert decisions and {e["action"] for e in decisions} == {"F"}
    assert {e["module"] for e in decisions} == {"L0.attn.F", "L0.ffn.F", "L1.attn.F", "L1.ffn.F"}
    # rank 4 / 2 teachers → 2 components per adapter → 2 × (d_in + d_out) coords
    ablation = [e for e in events if e["type"] == "ablation"]
    assert len(ablation) == 1 and ablation[0]["step"] == 5 and ablation[0]["teacher_index"] == 1
    assert ablation[0]["coords_zeroed"] > 0
    lora = torch.load(tmp_path / "ckpt" / "step_00000007" / "lora.pt", weights_only=False)
    # Teacher 1's slice (components 2,3) is dead after the ablation: A rows
    # and B columns stay exactly zero (zero A rows → zero gradients on the slice).
    for key, tensor in lora.items():
        if "lora_a" in key.lower():
            assert (tensor[2:4] == 0).all(), key
            assert (tensor[0:2] != 0).any(), key
        elif "lora_b" in key.lower():
            assert (tensor[:, 2:4] == 0).all(), key
    assert not [e for e in events if e["type"] == "consolidation"]

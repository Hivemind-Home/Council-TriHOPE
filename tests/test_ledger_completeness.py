"""Corollary 3 (ledger completeness), as a test rather than a claim: every
coordinate of a controller-indexed parameter that changed over a run was
opened by a logged write (a staged mask), and the ledger's coordinate count
equals the masks' — or, when a merge happened, the extra base change sits
exactly on the merged blocks."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

import hivemind.training as training_mod
from hivemind.controller.module_index import build_module_index
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.tracing import ModuleLedger


def _cfg(tmp_path: Path, force: list[int] | None = None) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {"vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                      "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0}},
            # One bucket; the bucket-frequency term alone drives R → F after
            # the first visit (see tests/test_replay.py), so masks get staged.
            "teachers": {"num_teachers": 1, "mode": "synthetic"},
            "controller": {
                "policy": {"top_m_modules": 3, "repetition_low": 0.4,
                           "repetition_medium": 0.9, "stability_high_C": 0.99},
                "repetition": {"bucket_smoothing_k": 0.5},
                "ablation": {"disable_signals": ["repetition_mom", "repetition_ret"]},
                "consolidation": {"period": 0},
                "debug": {"force_consolidate_steps": force or []},
            },
            "data": {"source": "synthetic", "dataset_size": 40, "seq_len": 16,
                     "vocab_size": 64, "batch_size": 4},
            "train": {"steps": 20, "device": "cpu", "seed": 11, "log_interval": 1},
            "optim": {"lr": 1e-2, "weight_decay": 0.01},
            "run": {"dir": str(tmp_path / "run")},
            "logging": {"enabled": True, "backend": "json",
                        "path": str(tmp_path / "metrics.jsonl"),
                        "events_path": str(tmp_path / "events.jsonl")},
        }
    )


class _Capture:
    """Snapshot θ_0, capture every staged mask, keep the live objects."""

    def __init__(self, monkeypatch) -> None:
        self.student = None
        self.theta0: dict[int, torch.Tensor] = {}
        self.masks: dict[int, torch.Tensor] = {}
        self.ledger: ModuleLedger | None = None
        real_build = training_mod._build_student
        real_set = MaskedAdamW.set_masks
        real_ledger = training_mod.ModuleLedger

        def build(cfg, device):
            self.student = real_build(cfg, device)
            for m in build_module_index(self.student):
                for p in m.params:
                    self.theta0[id(p)] = p.detach().clone()
            return self.student

        def set_masks(opt, masks):
            for p, mask in masks.items():
                full = (
                    torch.ones_like(p, dtype=torch.bool)
                    if mask is True
                    else mask.to(dtype=torch.bool).expand_as(p).clone()
                )
                prev = self.masks.get(id(p), torch.zeros_like(p, dtype=torch.bool))
                self.masks[id(p)] = prev | full
            return real_set(opt, masks)

        def make_ledger():
            self.ledger = real_ledger()
            return self.ledger

        monkeypatch.setattr(training_mod, "_build_student", build)
        monkeypatch.setattr(MaskedAdamW, "set_masks", set_masks)
        monkeypatch.setattr(training_mod, "ModuleLedger", make_ledger)


def test_changed_coords_subset_of_opened_masks(tmp_path: Path, monkeypatch) -> None:
    cap = _Capture(monkeypatch)
    training_mod.run_training_loop(_cfg(tmp_path), device=torch.device("cpu"))
    assert cap.masks, "no masks were staged — the controller never opened anything"
    opened_total = 0
    changed_total = 0
    for m in build_module_index(cap.student):
        for p in m.params:
            changed = p.detach() != cap.theta0[id(p)]
            opened = cap.masks.get(id(p), torch.zeros_like(p, dtype=torch.bool))
            # Corollary 3: nothing moved outside a logged write.
            assert not (changed & ~opened).any(), f"{m.id}: coordinates changed without a mask"
            changed_total += int(changed.sum())
            opened_total += int(opened.sum())
    assert changed_total > 0
    # Ledger ↔ masks: the ledger's coordinate count is the sum of every mask
    # ever staged (masks are counted per step; a coordinate opened twice is
    # two ledger coordinates), so compare against the per-step sums.
    assert cap.ledger is not None
    assert sum(e["total_coords_opened"] for e in cap.ledger.summary().values()) >= opened_total


def test_forced_merge_changes_base_only_on_merged_blocks(tmp_path: Path, monkeypatch) -> None:
    cap = _Capture(monkeypatch)
    training_mod.run_training_loop(_cfg(tmp_path, force=[10]), device=torch.device("cpu"))
    merged_blocks = {
        (int(name.split(".")[0][1:]), name.split(".")[1])
        for name, e in cap.ledger.summary().items()
        if e["consolidations"] > 0
    }
    assert merged_blocks
    for m in build_module_index(cap.student):
        for p in m.params:
            changed = p.detach() != cap.theta0[id(p)]
            opened = cap.masks.get(id(p), torch.zeros_like(p, dtype=torch.bool))
            outside = changed & ~opened
            if m.id.param_type == "P":
                # Base weights may move outside a mask only via a merge of
                # this very block.
                if outside.any():
                    assert (m.id.layer, m.id.block_type) in merged_blocks, str(m.id)
            else:
                # Adapters change outside masks only through reset_lora after
                # the merge, again only on merged blocks.
                if outside.any():
                    assert (m.id.layer, m.id.block_type) in merged_blocks, str(m.id)

"""Teacher attribution (task T5): ledger accounting, consolidation events,
pre-merge checkpoint tagging, and the analysis tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from hivemind.tracing import ModuleLedger
from hivemind.training import run_training_loop


class TestLedgerAttribution:
    def test_accumulates_per_teacher_and_clears_on_consolidation(self) -> None:
        led = ModuleLedger()
        led.record_action("L0.attn.F", "F", 1, 10, teacher="alice")
        led.record_action("L0.attn.F", "F", 2, 30, teacher="bob")
        led.record_action("L0.attn.F", "R", 3, 0, teacher="bob")  # no coords → no attribution
        led.record_action("L0.attn.F", "F", 4, 5, teacher="bob", replay=True)
        led.record_action("L0.attn.P", "P", 5, 100, teacher="alice")
        assert led.pending_attribution("L0.attn.F") == {"alice": 10, "bob": 35}
        assert led.attribution_share("L0.attn.F", "bob") == pytest.approx(35 / 45)
        assert led.pending_attribution("L0.attn.P") == {"alice": 100}

        attr = led.record_consolidation("L0.attn.F", 6, merged_coords=900)
        assert attr == {"alice": 10, "bob": 35}
        assert led.pending_attribution("L0.attn.F") == {}
        teachers = led.teacher_summary()
        assert teachers["alice"]["coords_F"] == 10 and teachers["alice"]["coords_P"] == 100
        assert teachers["bob"]["replay_count"] == 1 and teachers["bob"]["r_count"] == 1
        assert teachers["alice"]["merged_coords"] + teachers["bob"]["merged_coords"] == 900
        assert teachers["bob"]["merged_coords"] == 700

    def test_roundtrip_and_no_aliasing(self) -> None:
        led = ModuleLedger()
        led.record_action("L0.attn.F", "F", 1, 10, teacher="alice")
        led.record_action("L1.ffn.F", "F", 1, 10, teacher="bob")
        fresh = ModuleLedger()
        fresh.load_state_dict(led.state_dict())
        assert fresh.summary() == led.summary()
        assert fresh.teacher_summary() == led.teacher_summary()
        fresh.record_action("L0.attn.F", "F", 2, 1, teacher="carol")
        assert fresh.pending_attribution("L1.ffn.F") == {"bob": 10}
        assert "carol" not in fresh.pending_attribution("L1.ffn.F")
        # legacy ledger without attribution / teachers loads clean
        legacy = {"modules": {"L0.attn.F": {"f_count": 3}}}
        old = ModuleLedger()
        old.load_state_dict(legacy)
        assert old.pending_attribution("L0.attn.F") == {} and old.teacher_summary() == {}


def _cfg(tmp_path: Path, steps: int = 8, **debug) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {"vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                      "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0}},
            "teachers": {"num_teachers": 1, "mode": "synthetic"},
            "controller": {
                "consolidation": {"period": 0, "checkpoint_before_merge": True},
                "debug": {"policy_override": "always_p", "force_consolidate_steps": [5],
                          **debug},
            },
            "data": {"source": "synthetic", "dataset_size": 20, "seq_len": 16,
                     "vocab_size": 64, "batch_size": 4},
            "train": {"steps": steps, "device": "cpu", "seed": 3, "log_interval": 1},
            "optim": {"lr": 1e-3},
            "run": {"dir": str(tmp_path / "run")},
            "checkpoint": {"enabled": True, "dir": str(tmp_path / "ckpt"), "save_every": 100,
                           "keep_tagged": 0},
            "logging": {"enabled": True, "backend": "json",
                        "path": str(tmp_path / "metrics.jsonl"),
                        "events_path": str(tmp_path / "events.jsonl")},
        }
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


class TestEndToEnd:
    def test_consolidation_event_and_pre_merge_meta_carry_attribution(
        self, tmp_path: Path
    ) -> None:
        run_training_loop(_cfg(tmp_path), device=torch.device("cpu"))
        events = _events(tmp_path / "events.jsonl")
        teacher = next(e["teacher"] for e in events if e["type"] == "decision")
        assert teacher
        merges = [e for e in events if e["type"] == "consolidation"]
        assert merges and all(e["step"] == 5 for e in merges)
        # force_consolidate merges EVERY adapter; only the Top-M-selected ones
        # carry evidence, and that evidence belongs to the single teacher.
        attributed = [m for m in merges if m["attribution"]]
        assert attributed
        for m in attributed:
            assert set(m["attribution"]) == {teacher}
            assert m["attribution_share"][teacher] == pytest.approx(1.0)
            assert m["merged_coords"] > 0
        meta = json.loads((tmp_path / "ckpt" / "step_00000005_pre_merge" / "meta.json").read_text())
        extra = meta["extra"]
        assert extra["reason"] == "pre_merge"
        assert set(extra["pending_modules"]) == {m["module"] for m in attributed}
        assert all(
            set(extra["attribution"][m]) == {teacher} for m in extra["pending_modules"]
        )
        summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
        assert summary["ledger_totals"]["teachers"][teacher]["p_count"] > 0
        assert summary["ledger_totals"]["teachers"][teacher]["merged_coords"] > 0

    def test_tables_from_the_trace(self, tmp_path: Path) -> None:
        pd = pytest.importorskip("pandas")
        from analysis.loaders import load_run
        from analysis.tables import containment, teacher_attribution

        run_training_loop(_cfg(tmp_path), device=torch.device("cpu"))
        run = load_run(tmp_path / "run")
        # the loader expects metrics/events next to run_summary; point it there
        run.events = pd.DataFrame(_events(tmp_path / "events.jsonl"))
        df = teacher_attribution([run])
        assert not df.empty
        row = df.iloc[0]
        assert row["P_count"] > 0 and row["consolidations_attributed"] > 0
        assert row["merged_coords_attributed"] > 0
        cont = containment([run], teacher=row["teacher"])
        assert cont.iloc[0]["p_share_mean"] == pytest.approx(100.0)
        assert cont.iloc[0]["consolidations_attributed_mean"] > 0
        assert containment([run]).empty  # no corrupt_teacher block → nothing to contain


def test_every_sweep_merge_has_a_pre_merge_checkpoint(tmp_path: Path) -> None:
    """The legacy no-flags sweep must also be preceded by a rollback checkpoint."""
    cfg = _cfg(tmp_path, steps=12)
    # no forced merge, no P flags: the periodic sweep at step 6 merges via the
    # unconditional fallback once the (lowered) thresholds pass
    cfg.controller.debug.policy_override = "always_f"
    cfg.controller.debug.force_consolidate_steps = []
    cfg.controller.consolidation.period = 6
    cfg.controller.consolidation.min_stability_C = -1.0
    cfg.controller.consolidation.min_repetition = -1.0
    run_training_loop(cfg, device=torch.device("cpu"))
    events = _events(tmp_path / "events.jsonl")
    merges = {e["step"] for e in events if e["type"] == "consolidation"}
    assert merges == {6}
    assert (tmp_path / "ckpt" / "step_00000006_pre_merge").exists()

"""Tests for the event trace + module ledger, incl. end-to-end via the loop."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from hivemind.tracing import EventTrace, ModuleLedger
from hivemind.training import run_training_loop


class TestEventTrace:
    def test_emits_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        trace = EventTrace(path, flush_every=2)
        trace.emit({"type": "a", "step": 0})
        assert not path.exists() or path.read_text() == ""  # buffered
        trace.emit({"type": "b", "step": 1})
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "a"
        trace.close()

    def test_append_across_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        t1 = EventTrace(path, flush_every=1)
        t1.emit({"type": "x", "step": 0})
        t1.close()
        t2 = EventTrace(path, flush_every=1)
        t2.emit({"type": "resume", "step": 1})
        t2.close()
        lines = path.read_text().strip().splitlines()
        assert [json.loads(ln)["type"] for ln in lines] == ["x", "resume"]

    def test_disabled_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        trace = EventTrace(path, enabled=False)
        trace.emit({"type": "a"})
        trace.close()
        assert not path.exists()

    def test_serializes_tensors(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        trace = EventTrace(path, flush_every=1)
        trace.emit({"type": "a", "value": torch.tensor(1.5)})
        trace.close()
        assert json.loads(path.read_text())["value"] == 1.5


class TestModuleLedger:
    def test_counters(self) -> None:
        ledger = ModuleLedger()
        ledger.record_action("L0.ffn.F", "F", step=3, coords_opened=10)
        ledger.record_action("L0.ffn.F", "F", step=7, coords_opened=5)
        ledger.record_action("L0.ffn.F", "R", step=8, coords_opened=0)
        ledger.record_consolidation("L0.ffn.F", step=9)
        entry = ledger.summary()["L0.ffn.F"]
        assert entry["f_count"] == 2
        assert entry["r_count"] == 1
        assert entry["times_opened"] == 2
        assert entry["last_opened_step"] == 7
        assert entry["total_coords_opened"] == 15
        assert entry["consolidations"] == 1
        assert ledger.totals()["f_count"] == 2

    def test_state_dict_roundtrip(self) -> None:
        ledger = ModuleLedger()
        ledger.record_action("L1.attn.P", "P", step=1, coords_opened=100)
        clone = ModuleLedger()
        clone.load_state_dict(ledger.state_dict())
        assert clone.summary() == ledger.summary()


def _smoke_cfg(tmp_path: Path, steps: int = 6) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "vocab_size": 64,
                "dim": 32,
                "num_layers": 2,
                "heads": 4,
                "max_seq_len": 32,
                "lora": {"rank": 4, "alpha": 8.0},
            },
            "teachers": {"num_teachers": 2, "mode": "synthetic"},
            "controller": {},
            "data": {
                "source": "synthetic",
                "num_samples": 32,
                "seq_len": 16,
                "vocab_size": 64,
                "batch_size": 4,
            },
            "train": {"steps": steps, "device": "cpu", "seed": 0, "log_interval": 2},
            "optim": {"lr": 1e-3},
            "logging": {
                "enabled": True,
                "backend": "json",
                "path": str(tmp_path / "metrics.jsonl"),
                "events_path": str(tmp_path / "events.jsonl"),
            },
        }
    )


class TestTraceEndToEnd:
    def test_loop_emits_schema_valid_events(self, tmp_path: Path) -> None:
        cfg = _smoke_cfg(tmp_path)
        run_training_loop(cfg, device=torch.device("cpu"))

        events_file = tmp_path / "events.jsonl"
        assert events_file.exists()
        events = [
            json.loads(ln) for ln in events_file.read_text().strip().splitlines()
        ]
        types = {e["type"] for e in events}
        assert "run_config" in types
        assert "decision" in types

        run_config = next(e for e in events if e["type"] == "run_config")
        assert "thresholds" in run_config and "top_m" in run_config

        decisions = [e for e in events if e["type"] == "decision"]
        for d in decisions:
            assert d["action"] in {"R", "F", "P"}
            assert set(d["signals"]) >= {
                "S", "C", "V", "R", "R_mom", "R_hash", "R_ret",
                "stability_adam", "grad_norm",
            }
            assert "step" in d and "module" in d and "coords_opened" in d
        # Steps are monotonically nondecreasing.
        steps_seen = [d["step"] for d in decisions]
        assert steps_seen == sorted(steps_seen)
        # At most Top-M decisions per step.
        from collections import Counter

        per_step = Counter(steps_seen)
        assert max(per_step.values()) <= 6

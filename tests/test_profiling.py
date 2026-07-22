"""Tests for the profiling utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from hivemind.profiling import RunProfiler, StepTimer, write_run_summary


class TestStepTimer:
    def test_sections_accumulate(self) -> None:
        timer = StepTimer(ema_beta=0.5)
        for _ in range(3):
            with timer.section("work"):
                time.sleep(0.001)
        assert "work" in timer.ema()
        assert timer.ema()["work"] > 0
        assert timer.totals()["work"] >= 0.003


class TestRunProfiler:
    def test_step_metrics(self) -> None:
        prof = RunProfiler(device=torch.device("cpu"), ema_beta=0.5)
        for _ in range(2):
            prof.step_start()
            with prof.timer.section("signals"):
                time.sleep(0.001)
            with prof.timer.section("forward"):
                time.sleep(0.002)
            prof.step_end(tokens_in_step=128)
        metrics = prof.step_metrics()
        assert metrics["time/step_ema_s"] > 0
        assert metrics["time/tokens_per_sec"] > 0
        assert metrics["time/signals_ema_s"] > 0
        assert 0 <= metrics["time/controller_fraction"] <= 1

    def test_disabled_is_noop(self) -> None:
        prof = RunProfiler(device=torch.device("cpu"), enabled=False)
        prof.step_start()
        prof.step_end(tokens_in_step=10)
        assert prof.step_metrics() == {}

    def test_finalize_totals(self) -> None:
        prof = RunProfiler(device=torch.device("cpu"))
        prof.step_start()
        with prof.timer.section("policy_write"):
            time.sleep(0.001)
        prof.step_end(tokens_in_step=64)
        summary = prof.finalize()
        assert summary["steps"] == 1
        assert summary["tokens"] == 64
        assert summary["wall_clock_s"] > 0
        assert summary["controller_time_s"] > 0
        assert "section_totals_s" in summary


class TestRunSummary:
    def test_writes_json(self, tmp_path: Path) -> None:
        prof = RunProfiler(device=torch.device("cpu"))
        prof.step_start()
        prof.step_end(tokens_in_step=10)
        path = write_run_summary(
            tmp_path,
            profiler=prof,
            final_metrics={"loss/total": 1.5, "phase": "tail", "obj": object()},
            retention={"history": []},
            ledger_totals={"f_count": 3},
            interrupted=False,
            extra={"stream_digest": "abc123"},
        )
        payload = json.loads(path.read_text())
        assert payload["interrupted"] is False
        assert payload["profile"]["steps"] == 1
        assert payload["final_metrics"] == {"loss/total": 1.5, "phase": "tail"}
        assert payload["ledger_totals"]["f_count"] == 3
        assert payload["extra"]["stream_digest"] == "abc123"

    def test_interrupted_flag(self, tmp_path: Path) -> None:
        path = write_run_summary(tmp_path, interrupted=True)
        assert json.loads(path.read_text())["interrupted"] is True

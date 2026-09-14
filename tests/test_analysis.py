"""Tests for the analysis package against synthetic run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from analysis.loaders import load_experiment, load_run  # noqa: E402
from analysis.run_report import generate_report  # noqa: E402
from analysis.tables import (  # noqa: E402
    action_share_by_phase,
    damage_recovery,
    forgetting_table,
    p_selection_stats,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _make_run(root: Path, spec: str, seed: int, forced_step: int | None = None) -> Path:
    run_dir = root / f"{spec}-seed{seed}"
    run_dir.mkdir(parents=True)

    metrics = []
    for step in range(0, 100, 10):
        rec = {
            "step": step,
            "loss/total": 2.0 - step * 0.01,
            "phase": "warm" if step < 50 else "recurrent",
        }
        if step % 20 == 0:
            damaged = 0.5 if forced_step and step >= forced_step else 0.0
            rec["eval/macro_loss"] = 2.5 - step * 0.005 + damaged
            rec["eval/math/loss"] = 2.4 - step * 0.005
        metrics.append(rec)
    _write_jsonl(run_dir / "metrics.jsonl", metrics)

    events = [
        {"type": "run_config", "start_step": 0, "top_m": 4},
        {"type": "phase_start", "step": 0, "phase": "warm"},
        {"type": "phase_start", "step": 50, "phase": "recurrent"},
    ]
    for step in range(0, 100, 5):
        action = "R" if step < 50 else ("P" if step % 20 == 0 else "F")
        events.append(
            {
                "type": "decision", "step": step, "module": "L0.ffn.F",
                "action": action, "coords_opened": 8 if action != "R" else 0,
                "bucket_id": 1,
                "signals": {"S": 1.0, "C": 0.5, "V": 0.2, "R": 0.6, "grad_norm": 1.0},
            }
        )
    events.append(
        {"type": "consolidation", "step": 60, "module": "L0.ffn.F",
         "strategy": "direct", "forced": False, "success": True}
    )
    if forced_step is not None:
        events.append(
            {"type": "consolidation", "step": forced_step, "module": "L0.attn.F",
             "strategy": "direct", "forced": True, "success": True}
        )
    _write_jsonl(run_dir / "events.jsonl", events)

    (run_dir / "run_summary.json").write_text(json.dumps({
        "interrupted": False,
        "profile": {
            "wall_clock_s": 100.0, "tokens_per_sec": 500.0,
            "final_peak_mem_gb": 4.2, "controller_fraction": 0.1,
        },
        "retention": {"history": [
            {"phase": "warm", "step": 49, "loss": {"math": 2.0}, "em": {},
             "retention_delta": {}},
            {"phase": "recurrent", "step": 99, "loss": {"math": 2.2},
             "em": {"math": 0.25}, "retention_delta": {"math": 0.2}},
        ]},
        "ledger_totals": {"r_count": 10, "f_count": 6, "p_count": 4,
                          "consolidations": 1},
    }))
    return run_dir


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    root = tmp_path / "exp_v1"
    _make_run(root, "trihope", 1)
    _make_run(root, "trihope", 2)
    _make_run(root, "no_surprise", 1)
    _make_run(root, "p_forced_bad", 1, forced_step=60)
    return root


class TestLoaders:
    def test_load_run(self, experiment: Path) -> None:
        run = load_run(experiment / "trihope-seed1")
        assert run.spec_id == "trihope"
        assert run.seed == 1
        assert len(run.metrics) == 10
        assert run.phases == [(0, "warm"), (50, "recurrent")]
        assert run.phase_of_step(55) == "recurrent"

    def test_load_experiment(self, experiment: Path) -> None:
        runs = load_experiment(experiment)
        assert len(runs) == 4


class TestTables:
    def test_action_share_by_phase(self, experiment: Path) -> None:
        df = action_share_by_phase(load_experiment(experiment))
        warm = df[(df["spec_id"] == "trihope") & (df["phase"] == "warm")].iloc[0]
        assert warm["r_share_mean"] == 100.0
        rec = df[(df["spec_id"] == "trihope") & (df["phase"] == "recurrent")].iloc[0]
        assert rec["p_share_mean"] > 0

    def test_forgetting_table_aggregates_seeds(self, experiment: Path) -> None:
        df = forgetting_table(load_experiment(experiment))
        row = df[df["spec_id"] == "trihope"].iloc[0]
        assert row["final_loss_math_mean"] == pytest.approx(2.2)
        assert row["worst_retention_delta_math_mean"] == pytest.approx(0.2)
        assert row["peak_mem_gb_mean"] == pytest.approx(4.2)

    def test_p_selection_stats(self, experiment: Path) -> None:
        df = p_selection_stats(load_experiment(experiment))
        row = df[(df["spec_id"] == "trihope") & (df["seed"] == 1)].iloc[0]
        assert row["p_decisions"] == 2  # steps 60 and 80
        assert row["consolidations"] == 1

    def test_damage_recovery(self, experiment: Path) -> None:
        df = damage_recovery(load_experiment(experiment))
        assert len(df) == 1
        row = df.iloc[0]
        assert row["spec_id"] == "p_forced_bad"
        assert row["damage"] > 0


class TestReport:
    def test_generate_full_report(self, experiment: Path, tmp_path: Path) -> None:
        out = tmp_path / "analysis_out"
        report = generate_report(experiment, out)
        assert report.exists()
        text = report.read_text()
        assert "forgetting_table" in text
        assert (out / "forgetting_table.csv").exists()
        pngs = list(out.glob("*.png"))
        assert len(pngs) >= 3


def test_ablation_deltas_accepts_pooled_prefixed_baseline(tmp_path: Path) -> None:
    """``run_report A B`` prefixes spec ids with the experiment name; the
    ablation grid has no trihope of its own, so its deltas must resolve
    ``baselines_small_v1/trihope`` as the baseline."""
    from analysis.tables import ablation_deltas

    e1, grid = tmp_path / "baselines_small_v1", tmp_path / "ablation_grid_v1"
    _make_run(e1, "trihope", 1)
    _make_run(grid, "no_surprise", 1)
    runs = []
    for exp in (e1, grid):
        for run in load_experiment(exp):
            run.spec_id = f"{exp.name}/{run.spec_id}"
            run.run_id = f"{exp.name}/{run.run_id}"
            runs.append(run)
    df = ablation_deltas(runs)
    assert not df.empty
    assert list(df["spec_id"]) == ["ablation_grid_v1/no_surprise"]
    assert ablation_deltas(load_experiment(grid)).empty  # no baseline on its own


def test_replay_timing_splits_same_phase_from_cross_phase(tmp_path: Path) -> None:
    """Replays parked in one phase and written back in a later one are the
    stale-write-back case; same-phase ones are plain extra updates."""
    from analysis.tables import replay_timing

    run_dir = _make_run(tmp_path / "exp_v1", "trihope", 1)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    # phases: warm [0, 50), recurrent [50, ...). Two replays per event pair
    # (one per module) to check the per-module rows collapse to one replay.
    for step, origin in ((20, 10), (70, 60), (80, 30), (90, 30)):
        for module in ("L0.ffn.F", "L1.ffn.F"):
            events.append({"type": "replay", "step": step, "origin_step": origin,
                           "module": module, "action": "F", "coords_opened": 8,
                           "bucket_id": 7, "teacher": "t"})
    _write_jsonl(run_dir / "events.jsonl", events)
    df = replay_timing(load_experiment(tmp_path / "exp_v1"))
    row = df.iloc[0]
    assert row["replays"] == 4
    assert row["cross_phase_replays"] == 2 and abs(row["cross_phase_share"] - 0.5) < 1e-9
    assert row["replays_in_warm"] == 1 and row["replays_in_recurrent"] == 3
    assert row["delay_median_steps"] == 35.0  # delays 10, 10, 50, 60
    assert replay_timing(load_experiment(tmp_path / "exp_v1"))["seed"].iloc[0] == 1

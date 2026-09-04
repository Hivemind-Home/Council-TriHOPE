"""Budget-curve analysis (task T7): permanent writes vs. forgetting, the
steps-to-recover probe, the operating-point tag, and the Pareto figure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from analysis.figures import pareto_figure  # noqa: E402
from analysis.loaders import load_experiment, load_run  # noqa: E402
from analysis.run_report import generate_report  # noqa: E402
from analysis.tables import budget_curve, steps_to_recover, threshold_tag  # noqa: E402


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _run(
    root: Path,
    spec: str,
    seed: int,
    *,
    mode: str = "rfp",
    surprise_high: float = 2.0,
    permanent: int = 1000,
    recover_at: int | None = 50,
    override: str | None = None,
) -> Path:
    run_dir = root / f"{spec}-seed{seed}"
    run_dir.mkdir(parents=True)
    # phases: code_recurrent 0-99, math 100-199, code_revisit 200-299
    metrics = []
    for step in range(0, 300, 50):
        rec = {"step": step, "loss/total": 2.0, "phase": (
            "code_recurrent" if step < 100 else "math_recurrent" if step < 200 else "code_revisit"
        )}
        if step >= 200:
            recovered = recover_at is not None and step - 200 >= recover_at
            rec["eval/code/loss"] = 1.0 if recovered else 1.5
        metrics.append(rec)
    _jsonl(run_dir / "metrics.jsonl", metrics)
    events = [
        {
            "type": "run_config", "start_step": 0, "top_m": 4,
            "policy_mode": mode, "policy_override": override, "adam_score_high": 0.5,
            "thresholds": {"surprise_high": surprise_high, "repetition_low": 0.3},
            "consolidation": {"period": 250, "trigger": "signals", "plateau_tolerance": 0.01},
        },
        {"type": "phase_start", "step": 0, "phase": "code_recurrent"},
        {"type": "phase_start", "step": 100, "phase": "math_recurrent"},
        {"type": "phase_start", "step": 200, "phase": "code_revisit"},
    ]
    _jsonl(run_dir / "events.jsonl", events)
    (run_dir / "run_summary.json").write_text(json.dumps({
        "retention": {"history": [
            {"phase": "code_recurrent", "step": 99, "loss": {"code": 1.0}, "em": {},
             "retention_delta": {}},
            {"phase": "math_recurrent", "step": 199, "loss": {"code": 1.4, "math": 2.0},
             "em": {}, "retention_delta": {"code": 0.4}},
            {"phase": "code_revisit", "step": 299, "loss": {"code": 1.0, "math": 2.1},
             "em": {}, "retention_delta": {"math": 0.1}},
        ]},
        "ledger_totals": {"r_count": 1, "f_count": 1, "p_count": 1, "consolidations": 0},
        "permanent_writes": {"p_action_coords": permanent, "merged_coords": 0,
                             "unmasked_base_coords_total": 0, "total": permanent},
        "indexed_coords": 10_000,
        "active_fraction_mean": 0.12,
        "extra": {"retrieval": {"replayed_total": 7}},
    }))
    return run_dir


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    root = tmp_path / "e1"
    _run(root, "trihope", 1, permanent=1000)
    _run(root, "trihope", 2, permanent=3000)
    _run(root, "lora_only", 1, permanent=0, recover_at=None, mode="rfp")
    _run(root, "molf_style", 1, mode="adam_score", permanent=500_000)
    return root


class TestStepsToRecover:
    def test_recovers_after_50_steps(self, experiment: Path) -> None:
        run = load_run(experiment / "trihope-seed1")
        assert steps_to_recover(run) == 50

    def test_never_recovers_is_none(self, experiment: Path) -> None:
        run = load_run(experiment / "lora_only-seed1")
        assert steps_to_recover(run) is None


class TestThresholdTag:
    def test_tag_from_run_config(self, experiment: Path) -> None:
        assert threshold_tag(load_run(experiment / "trihope-seed1")) == "rfp_S2.0_R0.3"
        assert threshold_tag(load_run(experiment / "molf_style-seed1")) == "adam_score_A0.5"


class TestBudgetCurve:
    def test_columns_and_aggregation(self, experiment: Path) -> None:
        df = budget_curve(load_experiment(experiment))
        row = df[df["spec_id"] == "trihope"].iloc[0]
        assert row["permanent_writes_mean"] == pytest.approx(2000.0)
        assert row["permanent_writes_std"] == pytest.approx(1414.2135, rel=1e-3)
        assert row["worst_retention_delta_mean"] == pytest.approx(0.4)
        assert row["mean_retention_delta_mean"] == pytest.approx(0.25)
        assert row["final_loss_math_mean"] == pytest.approx(2.1)
        assert row["steps_to_recover_code_mean"] == pytest.approx(50)
        assert row["replayed_total_mean"] == pytest.approx(7)
        assert row["active_fraction_mean_mean"] == pytest.approx(0.12)
        lora = df[df["spec_id"] == "lora_only"].iloc[0]
        assert lora["permanent_writes_mean"] == 0
        assert pd.isna(lora["steps_to_recover_code_mean"])

    def test_pareto_figure_handles_zero_writes(self, experiment: Path, tmp_path: Path) -> None:
        df = budget_curve(load_experiment(experiment))
        out = pareto_figure(df, tmp_path / "pareto.png")
        assert out is not None and out.exists() and out.stat().st_size > 0
        assert pareto_figure(pd.DataFrame(), tmp_path / "none.png") is None


class TestMultiDirReport:
    def test_two_experiments_are_prefixed(self, experiment: Path, tmp_path: Path) -> None:
        sweep = tmp_path / "sweep"
        _run(sweep, "trihope", 1, surprise_high=4.0, permanent=200)
        report = generate_report([experiment, sweep], tmp_path / "out")
        text = report.read_text()
        assert "e1/trihope" in text and "sweep/trihope" in text
        assert (tmp_path / "out" / "budget_curve.csv").exists()
        assert (tmp_path / "out" / "pareto_budget.png").exists()
        df = pd.read_csv(tmp_path / "out" / "budget_curve.csv")
        assert set(df["spec_id"]) >= {"e1/trihope", "sweep/trihope"}

    def test_single_dir_path_still_works(self, experiment: Path, tmp_path: Path) -> None:
        report = generate_report(experiment, tmp_path / "out1")
        assert report.exists()

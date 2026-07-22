"""Load per-run artifacts (metrics, events, summary) into DataFrames."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_RUN_ID_RE = re.compile(r"^(?P<spec>.+)-seed(?P<seed>\d+)$")


@dataclass
class RunData:
    run_id: str
    spec_id: str
    seed: int
    run_dir: Path
    metrics: pd.DataFrame          # one row per logged step
    events: pd.DataFrame           # one row per trace event
    summary: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)

    @property
    def phases(self) -> list[tuple[int, str]]:
        """(start_step, phase_name) from phase_start events."""
        if self.events.empty:
            return []
        starts = self.events[self.events["type"] == "phase_start"]
        seen: dict[str, int] = {}
        for _, row in starts.iterrows():
            seen.setdefault(row["phase"], int(row["step"]))
        return sorted(((s, p) for p, s in seen.items()))

    def phase_of_step(self, step: int) -> Optional[str]:
        name = None
        for start, phase in self.phases:
            if step >= start:
                name = phase
        return name


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(records)


def load_run(run_dir: "str | Path") -> RunData:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    m = _RUN_ID_RE.match(run_id)
    spec_id, seed = (m.group("spec"), int(m.group("seed"))) if m else (run_id, 0)

    summary_path = run_dir / "run_summary.json"
    status_path = run_dir / "status.json"
    return RunData(
        run_id=run_id,
        spec_id=spec_id,
        seed=seed,
        run_dir=run_dir,
        metrics=_read_jsonl(run_dir / "metrics.jsonl"),
        events=_read_jsonl(run_dir / "events.jsonl"),
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {},
        status=json.loads(status_path.read_text()) if status_path.exists() else {},
    )


def load_experiment(exp_dir: "str | Path") -> list[RunData]:
    """Load every run directory under an experiment root."""
    exp_dir = Path(exp_dir)
    runs = []
    for child in sorted(exp_dir.iterdir()):
        if child.is_dir() and (
            (child / "metrics.jsonl").exists() or (child / "run_summary.json").exists()
        ):
            runs.append(load_run(child))
    if not runs:
        raise FileNotFoundError(f"No run directories with artifacts under {exp_dir}")
    return runs

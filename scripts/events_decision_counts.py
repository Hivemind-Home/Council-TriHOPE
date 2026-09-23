#!/usr/bin/env python
"""Count controller decisions by action x routed surface x phase (paper T2).

The paper describes each decision by what it actually does to the weights:
a P decision on a base block commits it, an F decision on a base block
opens nothing (withhold), an F decision on an adapter opens Top-K ranks,
R parks the example. That split needs ``events.jsonl`` (not in git), so
this script reduces each run's trace to two small files the paper repo
can snapshot:

  <run>/analysis_t2/decisions.csv  phase, action, surface, count, coords_opened
  <run>/analysis_t2/summary.json   decisions by (action, surface), replay writes,
                                   merges nominal vs effective (+ merged coords)

A merge is *effective* when the merged adapter received at least one write
(a replay micro-step or a direct F/P decision that opened coordinates on it)
since its last reset; otherwise it re-initialises an all-zero adapter and
changes nothing, although the ledger counts its base size as permanent writes.

    python scripts/events_decision_counts.py runs/*_live/*-seed*
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def surface(module: str) -> str:
    # ModuleId str: L{layer}.{attn|ffn}.{P|F}
    return "adapter" if module.endswith(".F") else "base"


def reduce_run(run: Path) -> dict | None:
    ev_path = run / "events.jsonl"
    if not ev_path.exists():
        return None
    phase = "pre"
    counts: Counter = Counter()
    coords: Counter = Counter()
    replay_writes: Counter = Counter()
    written_since_reset: dict[str, bool] = defaultdict(bool)
    merges = {"nominal": 0, "effective": 0, "forced": 0}
    merged_coords = {"nominal": 0, "effective": 0}
    with ev_path.open() as fh:
        for line in fh:
            if '"type": "decision"' not in line and '"type": "replay"' not in line \
                    and '"type": "consolidation"' not in line and '"type": "phase_start"' not in line:
                continue
            ev = json.loads(line)
            kind = ev.get("type")
            if kind == "phase_start":
                phase = ev.get("phase", phase)
            elif kind == "decision":
                mod, act = str(ev["module"]), str(ev["action"])
                key = (phase, act, surface(mod))
                counts[key] += 1
                coords[key] += int(ev.get("coords_opened", 0))
                if int(ev.get("coords_opened", 0)) > 0 and surface(mod) == "adapter":
                    written_since_reset[mod] = True
            elif kind == "replay":
                mod = str(ev["module"])
                replay_writes[(phase, surface(mod))] += 1
                if int(ev.get("coords_opened", 0)) > 0:
                    written_since_reset[mod] = True
            elif kind == "consolidation":
                mod = str(ev["module"])
                mc = int(ev.get("merged_coords", 0))
                merges["nominal"] += 1
                merged_coords["nominal"] += mc
                if ev.get("forced") or ev.get("trigger") == "forced":
                    merges["forced"] += 1
                if written_since_reset[mod]:
                    merges["effective"] += 1
                    merged_coords["effective"] += mc
                written_since_reset[mod] = False

    out_dir = run / "analysis_t2"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "action", "surface", "count", "coords_opened"])
        for (ph, act, surf), n in sorted(counts.items()):
            w.writerow([ph, act, surf, n, coords[(ph, act, surf)]])
    by_action: Counter = Counter()
    for (_, act, surf), n in counts.items():
        by_action[f"{act}_{surf}"] += n
    replay_by_surface: Counter = Counter()
    for (_, surf), n in replay_writes.items():
        replay_by_surface[surf] += n
    controller_on = sum(counts.values()) > 0
    if not controller_on:
        # Controller off (lora_only / plateau / periodic_merge): every adapter
        # trains on every step without decision events, so every merge moves
        # a trained adapter into base.
        merges["effective"] = merges["nominal"]
        merged_coords["effective"] = merged_coords["nominal"]
    summary = {
        "run": run.name,
        "controller_decisions": int(sum(counts.values())),
        "decisions_by_action_surface": dict(sorted(by_action.items())),
        "replay_writes_by_surface": dict(replay_by_surface),
        "replay_writes_total": int(sum(replay_writes.values())),
        "merges": merges,
        "merged_coords": merged_coords,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for run in args.runs:
        if not run.is_dir():
            continue
        s = reduce_run(run)
        if s is None:
            print(f"{run}: no events.jsonl, skipped")
            continue
        print(
            f"{run}: {s['decisions_by_action_surface']} replay={s['replay_writes_total']} "
            f"merges={s['merges']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

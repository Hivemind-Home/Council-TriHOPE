"""Manipulation check: mean agreement (C-bar) of the blocks each arm actually committed.

Reads the --json files written by commit_schedule.py and prints, per file, the
commit-weighted mean C-bar of the reference and of every arm. A schedule_anti arm
must commit lower-agreement blocks than its reference, or the control did not work.
Usage: python scripts/commit_cbar.py runs/rebuttal_volume_live/commit_schedule_*_seed*.json
"""
import json
import sys


def weighted_cbar(summary: dict) -> float:
    phases = [p for p in summary["phases"].values()
              if p["commits"] and p.get("mean_C_bar") is not None]
    return sum(p["mean_C_bar"] * p["commits"] for p in phases) / sum(p["commits"] for p in phases)


def main(paths: list[str]) -> None:
    for path in sorted(paths):
        d = json.load(open(path))
        cols = [f"reference {weighted_cbar(d['reference_summary']):.3f}"]
        cols += [f"{run.rsplit('/', 1)[-1]} {weighted_cbar(s):.3f}" for run, s in d["runs"].items()]
        print(f"{path.rsplit('/', 1)[-1]}: " + " | ".join(cols))


if __name__ == "__main__":
    main(sys.argv[1:])

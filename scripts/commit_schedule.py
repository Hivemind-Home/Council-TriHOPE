#!/usr/bin/env python
"""Commit schedules from ``commits.jsonl``: timing, volume, and overlap.

Reads the per-step commit trace that every run writes next to its
``events.jsonl`` (one record per trained step: R/F/P counts; on committing
steps also the committed blocks and their agreement; on yoked arms also the
blocks the controller itself would have committed) and reports, per run and
per phase:

* commits (P labels), deferrals (R) and F writes;
* commit timing inside the phase: the fraction of commits in the first
  ``--early-window`` steps, and a decile histogram of steps-into-phase
  (reviewer 1Ume: "show the random arm's commit-time distribution");
* for yoked arms: how many of the arm's commits landed on a block the
  controller would have committed that step (overlap), and the controller's
  own commit count on the same trajectory (must equal the arm's — that is
  the live-yoking invariant, checked here);
* against ``--reference``: the ratio of commit totals per phase and overall.
  With ``--expect-match`` the script exits 1 when any phase with at least
  ``--min-commits`` reference commits is outside ``1 ± --tolerance``.

    python scripts/commit_schedule.py runs/rebuttal_v1_live/yoked_random-seed1337 \\
        --reference runs/rebuttal_v1_live/trihope_e256-seed1337 --expect-match
    python scripts/commit_schedule.py runs/rebuttal_v1_live/*-seed* --json commit_schedule.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(run: Path) -> list[dict]:
    path = run / "commits.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{run}: no commits.jsonl (run predates the trace, or logging off)")
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # A resumed run appends; keep the last record per step.
    by_step: dict[int, dict] = {}
    for r in recs:
        by_step[int(r["step"])] = r
    return [by_step[s] for s in sorted(by_step)]


def summarize(recs: list[dict], early: int) -> dict:
    phases: dict[str, dict] = {}
    order: list[str] = []
    start: dict[str, int] = {}
    end: dict[str, int] = {}
    for r in recs:
        ph = r.get("phase") or "_"
        if ph not in phases:
            order.append(ph)
            start[ph] = r["step"]
            phases[ph] = {
                "steps": 0, "commits": 0, "commit_steps": 0, "defers": 0, "f_writes": 0,
                "early_commits": 0, "decile_hist": [0] * 10, "overlap_hits": 0,
                "controller_commits": 0, "C_bar_sum": 0.0,
            }
        end[ph] = r["step"]
        p = phases[ph]
        p["steps"] += 1
        p["defers"] += int(r.get("r", 0))
        p["f_writes"] += int(r.get("f", 0))
        k = int(r.get("p", 0))
        if k:
            p["commits"] += k
            p["commit_steps"] += 1
            into = r["step"] - start[ph]
            if into < early:
                p["early_commits"] += k
            p["C_bar_sum"] += sum(r.get("C_bar_p", []))
        ctl = r.get("controller_p")
        if ctl is not None:
            p["controller_commits"] += len(ctl)
            p["overlap_hits"] += len(set(ctl) & set(r.get("modules_p", [])))
    # Decile histogram needs the phase length, so a second pass.
    for r in recs:
        k = int(r.get("p", 0))
        if not k:
            continue
        ph = r.get("phase") or "_"
        length = max(1, end[ph] - start[ph] + 1)
        d = min(9, int(10 * (r["step"] - start[ph]) / length))
        phases[ph]["decile_hist"][d] += k
    out = {"phases": {}, "total": {}}
    tot = defaultdict(float)
    for ph in order:
        p = phases[ph]
        row = {
            "start": start[ph], "end": end[ph], "steps": p["steps"],
            "commits": p["commits"], "commit_steps": p["commit_steps"],
            "defers": p["defers"], "f_writes": p["f_writes"],
            "early_frac": (p["early_commits"] / p["commits"]) if p["commits"] else None,
            "decile_hist": p["decile_hist"],
            "mean_C_bar": (p["C_bar_sum"] / p["commits"]) if p["commits"] else None,
        }
        if p["controller_commits"] or any(("controller_p" in r) for r in recs):
            row["controller_commits"] = p["controller_commits"]
            row["overlap_frac"] = (p["overlap_hits"] / p["commits"]) if p["commits"] else None
        out["phases"][ph] = row
        for key in ("steps", "commits", "commit_steps", "defers", "f_writes", "early_commits",
                    "controller_commits", "overlap_hits"):
            tot[key] += p[key]
    out["total"] = {
        "steps": int(tot["steps"]), "commits": int(tot["commits"]),
        "commit_steps": int(tot["commit_steps"]), "defers": int(tot["defers"]),
        "f_writes": int(tot["f_writes"]),
        "early_frac": (tot["early_commits"] / tot["commits"]) if tot["commits"] else None,
    }
    if tot["controller_commits"]:
        out["total"]["controller_commits"] = int(tot["controller_commits"])
        out["total"]["overlap_frac"] = tot["overlap_hits"] / tot["commits"] if tot["commits"] else None
    return out


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="run whose commit volume the others are compared with")
    ap.add_argument("--early-window", type=int, default=200,
                    help="steps after a phase start that count as 'early' (default 200)")
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--min-commits", type=int, default=5,
                    help="phases with fewer reference commits are not volume-checked")
    ap.add_argument("--expect-match", action="store_true",
                    help="exit 1 if any checked phase is outside 1 ± tolerance of the reference")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    ref = summarize(load(args.reference), args.early_window) if args.reference else None
    report: dict[str, dict] = {}
    failed = False
    for run in args.runs:
        s = summarize(load(run), args.early_window)
        report[str(run)] = s
        print(f"\n== {run}")
        print(f"{'phase':<20}{'steps':>7}{'commits':>9}{'c-steps':>9}{'defers':>8}"
              f"{'early':>7}{'C̄':>7}{'ctl':>6}{'ovl':>6}{'vs ref':>8}  deciles")
        for ph, row in s["phases"].items():
            ratio = None
            if ref and ph in ref["phases"] and ref["phases"][ph]["commits"]:
                ratio = row["commits"] / ref["phases"][ph]["commits"]
                if (
                    args.expect_match
                    and ref["phases"][ph]["commits"] >= args.min_commits
                    and abs(ratio - 1.0) > args.tolerance
                ):
                    failed = True
            print(
                f"{ph:<20}{row['steps']:>7}{row['commits']:>9}{row['commit_steps']:>9}"
                f"{row['defers']:>8}{fmt(row['early_frac']):>7}{fmt(row['mean_C_bar']):>7}"
                f"{fmt(row.get('controller_commits')):>6}{fmt(row.get('overlap_frac')):>6}"
                f"{fmt(ratio):>8}  {row['decile_hist']}"
            )
            row["ratio_vs_reference"] = ratio
        t = s["total"]
        total_ratio = (t["commits"] / ref["total"]["commits"]) if ref and ref["total"]["commits"] else None
        t["ratio_vs_reference"] = total_ratio
        print(
            f"{'TOTAL':<20}{t['steps']:>7}{t['commits']:>9}{t['commit_steps']:>9}{t['defers']:>8}"
            f"{fmt(t['early_frac']):>7}{'':>7}{fmt(t.get('controller_commits')):>6}"
            f"{fmt(t.get('overlap_frac')):>6}{fmt(total_ratio):>8}"
        )
        if "controller_commits" in t and t["controller_commits"] != t["commits"]:
            # Live-yoking invariant: the arm commits exactly as often as the
            # controller would have on the same trajectory.
            print(f"   FAIL: yoked arm committed {t['commits']} but the controller would have "
                  f"committed {t['controller_commits']} on this trajectory")
            failed = True
        if args.expect_match and total_ratio is not None and abs(total_ratio - 1.0) > args.tolerance:
            failed = True
    if args.json:
        args.json.write_text(json.dumps(
            {"reference": str(args.reference) if args.reference else None,
             "reference_summary": ref, "runs": report}, indent=1))
    if failed:
        print("\nFAIL: commit volume outside tolerance (see above)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

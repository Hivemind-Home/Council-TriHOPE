#!/usr/bin/env python
"""Manipulation check for the schedule_oracle_* arms (from commits.jsonl).

For every committing step the oracle arms log ``score_p`` (oracle score of each
committed block: first-order predicted change in earlier-domain loss) and
``score_cand_mean`` (mean over all non-deferred candidates, i.e. what a random
pick gets in expectation), plus ``oracle_fallback`` (first phase: no earlier
domain yet, so the pick was schedule_random's).

Per run this prints: committing steps, the fallback fraction, the mean
committed score vs the mean candidate score on scored steps, and the fraction
of scored steps where the committed mean lies on the intended side of the
candidate mean (bad: above, good: below; ties count as neither). The check
passes when that fraction is >= --min-side for every run (exit 1 otherwise).

    python scripts/oracle_check.py runs/rebuttal_oracle_live/*oracle*-seed*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--min-side", type=float, default=0.9)
    args = ap.parse_args(argv)
    ok = True
    print(f"{'run':<44}{'c-steps':>8}{'fallbk':>8}{'scored':>8}{'mean score_p':>14}"
          f"{'mean cand':>12}{'right side':>12}")
    for run in args.runs:
        by_step: dict[int, dict] = {}
        for line in (run / "commits.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_step[int(r["step"])] = r  # keep-last under resume
        com = [r for r in by_step.values() if r.get("p")]
        bad = "oracle_bad" in run.name
        fb = sum(1 for r in com if r.get("oracle_fallback"))
        scored = [r for r in com if not r.get("oracle_fallback") and r.get("score_p")
                  and "score_cand_mean" in r]
        sp = [sum(r["score_p"]) / len(r["score_p"]) for r in scored]
        cm = [r["score_cand_mean"] for r in scored]
        side = [(a > b) if bad else (a < b) for a, b in zip(sp, cm) if a != b]
        frac = (sum(side) / len(side)) if side else None
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
        print(f"{run.name:<44}{len(com):>8}{fb:>8}{len(scored):>8}{mean(sp):>14.3e}"
              f"{mean(cm):>12.3e}{'-' if frac is None else f'{frac:.2f}':>12}")
        if frac is None or frac < args.min_side:
            ok = False
    print("manipulation check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

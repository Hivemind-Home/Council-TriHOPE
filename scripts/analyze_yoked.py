#!/usr/bin/env python3
"""Read the replay-yoked 2x2 against its pre-registered rule.

Rule (fixed in configs/experiments/yoked_2x2.yaml before any run started):
  full wins at BOTH fixed budgets      -> allocation carries information
  only budget moves the metric         -> the signals set volume, not choice
  neither moves it                     -> shared-parameter learning dominates

Usage: python scripts/analyze_yoked.py runs/yoked_2x2_v1_live
"""
import json, sys, statistics as st
from pathlib import Path

DOM = ["general", "code", "math", "medical"]
ARMS = ["yoked_full_lo", "yoked_full_hi", "yoked_blind_lo", "yoked_blind_hi"]


def load(root: Path):
    out = {}
    for d in sorted(root.glob("*-seed*")):
        spec = d.name.rsplit("-seed", 1)[0]
        try:
            s = json.load(open(d / "run_summary.json"))
        except Exception:
            continue
        # retention.history is a LIST of per-phase records, each with a
        # {domain: loss} map -- not a dict of series. Recomputed here so this
        # works before analysis/ exists: per domain, the worst rise above the
        # best loss seen so far, which is what worst_retention_delta means.
        hist = (s.get("retention") or {}).get("history") or []
        wd = []
        for dom in DOM:
            best = None
            worst = 0.0
            seen = False
            for rec in hist:
                v = (rec.get("loss") or {}).get(dom)
                if v is None:
                    continue
                seen = True
                if best is None or v < best:
                    best = v
                worst = max(worst, v - best)
            if seen:
                wd.append(worst)
        acq = (s.get("retention") or {}).get("own_phase_loss") or {}
        led = s.get("ledger_totals") or {}
        out.setdefault(spec, []).append({
            "forget": st.mean(wd) if wd else None,
            "acq": st.mean(acq.values()) if acq else None,
            "replay": led.get("replay_count", 0),
            "coords": led.get("coords_opened", 0),
        })
    return out


def main(root):
    runs = load(Path(root))
    if not runs:
        print(f"no completed runs under {root}")
        return 1
    print(f"{'arm':<18}{'n':>2}{'forget':>10}{'acq':>10}{'replay':>9}{'coords/1e6':>12}")
    print("-" * 61)
    stat = {}
    for a in ARMS:
        rs = [r for r in runs.get(a, []) if r["forget"] is not None]
        if not rs:
            print(f"{a:<18}{0:>2}{'  pending':>10}")
            continue
        f = [r["forget"] for r in rs]
        stat[a] = f
        print(f"{a:<18}{len(rs):>2}{st.mean(f):>10.4f}"
              f"{st.mean([r['acq'] for r in rs if r['acq']]):>10.4f}"
              f"{st.mean([r['replay'] for r in rs]):>9.0f}"
              f"{st.mean([r['coords'] for r in rs])/1e6:>12.1f}")
    print()
    if not all(k in stat for k in ARMS):
        print("2x2 incomplete -- verdict withheld until all four cells have data.")
        return 0
    def sd(x):
        return st.stdev(x) if len(x) > 1 else float("nan")
    lo = st.mean(stat["yoked_blind_lo"]) - st.mean(stat["yoked_full_lo"])
    hi = st.mean(stat["yoked_blind_hi"]) - st.mean(stat["yoked_full_hi"])
    noise = max(sd(stat["yoked_full_lo"]), sd(stat["yoked_full_hi"]), 0.006)
    budget = (st.mean(stat["yoked_full_hi"]) + st.mean(stat["yoked_blind_hi"])) / 2 - \
             (st.mean(stat["yoked_full_lo"]) + st.mean(stat["yoked_blind_lo"])) / 2
    print(f"blind - full at LOW budget : {lo:+.4f}")
    print(f"blind - full at HIGH budget: {hi:+.4f}")
    print(f"budget effect (hi - lo)    : {budget:+.4f}")
    print(f"noise floor used           : {noise:.4f}\n")
    alloc = lo > noise and hi > noise
    vol = abs(budget) > noise
    if alloc:
        print("VERDICT: allocation carries information (blind is worse at both budgets).")
    elif vol:
        print("VERDICT: only replay volume moves the metric; the signals set volume, not choice.")
    else:
        print("VERDICT: neither allocation nor budget moves it; shared-parameter learning dominates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/yoked_2x2_v1_live"))

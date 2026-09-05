"""Why did P never fire? Signal distributions at decision time, per phase.

    python scripts/diagnose_p.py runs/<exp>/<run> [--config path/to/.hydra/config.yaml]

Reads events.jsonl: maps each decision to its phase via the phase_start
events, then reports per phase the quantiles of R, C̄, V and S, the share
of decisions passing each P sub-condition (policy thresholds) and both
sub-conditions together, and the same for the consolidation sweep's
re-validation (min_repetition / min_stability_C) at the sweep steps. The
run_config event supplies the thresholds when present.
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np


def _q(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    a = np.asarray(xs, dtype=float)
    return "p10 %.3f  p50 %.3f  p90 %.3f  max %.3f" % tuple(np.percentile(a, [10, 50, 90, 100]))


def main(run_dir: str, config_path: str | None = None) -> int:
    run = Path(run_dir)
    ev = [json.loads(line) for line in open(run / "events.jsonl")]
    ctrl: dict = {}
    snapshots = tuple(
        [Path(config_path)] if config_path else []
    ) + (
        run / "hydra" / "rank0" / ".hydra" / "config.yaml",
        run / ".hydra" / "config.yaml",
    )
    for cand in snapshots:
        if cand.exists():
            import yaml  # hydra's resolved config snapshot, written by the runner

            ctrl = (yaml.safe_load(cand.read_text()) or {}).get("controller") or {}
            print(f"thresholds from {cand}")
            break
    else:
        print("thresholds: no hydra snapshot under the run dir; using stream_small defaults")
    pol = ctrl.get("policy") or {}
    con = ctrl.get("consolidation") or {}
    rep_med = float(pol.get("repetition_medium", 0.5))
    c_high = float(pol.get("stability_high_C", 0.5))
    v_low = float(pol.get("stability_low_V", 0.3))
    min_rep = float(con.get("min_repetition", 0.4))
    min_c = float(con.get("min_stability_C", 0.4))
    period = int(con.get("period", 250) or 0)
    src = pol.get("stability_source", "sustained")
    print(f"policy P needs: R >= {rep_med}, C({src}) >= {c_high}, V <= {v_low}")
    dis = (ctrl.get("ablation") or {}).get("disable_stores") or []
    if "P" in dis:
        print("NOTE: P is disabled by controller.ablation.disable_stores for this run")
    print(f"sweep merge needs: R >= {min_rep}, C_bar >= {min_c}, every {period} steps\n")

    starts = sorted((e["step"], e["phase"]) for e in ev if e.get("type") == "phase_start")
    steps = [s for s, _ in starts]

    def phase_of(step: int) -> str:
        i = bisect_right(steps, step) - 1
        return starts[i][1] if i >= 0 else "?"

    by_phase: dict[str, list[dict]] = {}
    for e in ev:
        if e.get("type") != "decision":
            continue
        by_phase.setdefault(phase_of(e["step"]), []).append(e)

    for phase, decs in by_phase.items():
        sig = [d["signals"] for d in decs]
        R = [x["R"] for x in sig]
        C = [x["C_bar" if src == "sustained" else "C"] for x in sig]
        V = [x["V"] for x in sig]
        S = [x["S"] for x in sig]
        ok_r = np.mean([r >= rep_med for r in R])
        ok_c = np.mean([c >= c_high for c in C])
        ok_v = np.mean([v <= v_low for v in V])
        ok_all = np.mean([r >= rep_med and c >= c_high and v <= v_low for r, c, v in zip(R, C, V)])
        acts = {a: sum(d["action"] == a for d in decs) for a in ("R", "F", "P")}
        print(f"== {phase}  decisions={len(decs)}  actions={acts}")
        print(f"   R     {_q(R)}   pass {ok_r:.2%}")
        print(f"   C     {_q(C)}   pass {ok_c:.2%}")
        print(f"   V     {_q(V)}   pass {ok_v:.2%}")
        print(f"   S     {_q(S)}")
        print(f"   all three at once: {ok_all:.2%}")
        if period:
            sweep = [x for d, x in zip(decs, sig) if d["step"] % period == 0]
            if sweep:
                ok = np.mean([x["R"] >= min_rep and x["C_bar"] >= min_c for x in sweep])
                print(
                    f"   at sweep steps ({len(sweep)} decisions): "
                    f"R>={min_rep} & C_bar>={min_c} -> {ok:.2%}"
                )
    cons = [e for e in ev if e.get("type") == "consolidation"]
    print(f"\nconsolidation events: {len(cons)}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=None, help="hydra config.yaml (defaults: under run_dir)")
    ns = ap.parse_args()
    sys.exit(main(ns.run_dir, ns.config))

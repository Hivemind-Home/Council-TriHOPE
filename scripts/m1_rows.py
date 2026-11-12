"""M1 (worst retention delta) from eval_rows.jsonl on the first n rows per domain.

Replicates PhaseEvalTracker.record: delta_d = loss_d(now) - loss_d(end of last phase that
trained d), at phase boundaries. Validated by requiring the 256-row result to match the
run_summary.json retention history exactly.
Usage: python scripts/m1_rows.py results_live/rebuttal_v1_live [results_live/rebuttal_volume_live]
"""
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOTS = [pathlib.Path(a) for a in sys.argv[1:]]
SEEDS = ("1337", "2024", "7")


def phase_losses(run, n):
    """{step: {domain: loss on first n rows}} for every eval step (last record wins)."""
    out = defaultdict(dict)
    for line in (run / "eval_rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        ls, tk = np.array(r["loss_sum"][:n]), np.array(r["tokens"][:n])
        out[r["step"]][r["domain"]] = ls.sum() / tk.sum()
    return out


def deltas(run, n):
    hist = json.loads((run / "run_summary.json").read_text())["retention"]["history"]
    rows = phase_losses(run, n)
    own, worst, stored = {}, {}, {}
    for e in hist:
        loss = rows[e["step"]]
        # phase domains are exactly the domains the tracker did not report a delta for
        phase_domains = [d for d in loss if d not in e["retention_delta"]]
        for d, cur in loss.items():
            if d in own and d not in phase_domains:
                worst[d] = max(worst.get(d, -np.inf), cur - own[d])
        for d, v in e["retention_delta"].items():
            stored[d] = max(stored.get(d, -np.inf), v)
        for d in phase_domains:
            own[d] = loss[d]
    return worst, stored


def main():
    where = {p.name.rsplit("-seed", 1)[0]: r
             for r in ROOTS for p in r.glob("*-seed*") if p.is_dir()}
    arms = sorted(where)
    bad = 0
    res = {}
    for arm in arms:
        for s in SEEDS:
            run = where[arm] / f"{arm}-seed{s}"
            w256, stored = deltas(run, 256)
            for d in stored:  # validation: 256-row recompute must equal the logged deltas
                if abs(w256[d] - stored[d]) > 1e-6:
                    bad += 1
                    print(f"MISMATCH {run.name} {d}: {w256[d]:.6f} vs stored {stored[d]:.6f}")
            w32, _ = deltas(run, 32)
            # paper M1: mean over domains of the worst retention delta
            res[(arm, s)] = (np.mean(list(w32.values())), np.mean(list(w256.values())))
    print(f"validation mismatches: {bad}\n")
    print(f"{'arm':<28} {'M1@32 per seed (1337/2024/7)':<30} {'mean32':>7} "
          f"{'M1@256 per seed':<26} {'mean256':>7}")
    for arm in arms:
        a32 = [res[(arm, s)][0] for s in SEEDS]
        a256 = [res[(arm, s)][1] for s in SEEDS]
        print(f"{arm:<28} {' / '.join(f'{x:.3f}' for x in a32):<30} {np.mean(a32):7.3f} "
              f"{' / '.join(f'{x:.3f}' for x in a256):<26} {np.mean(a256):7.3f}")
    print("\npaired differences (per-run M1@256, arm - ref, per seed):")
    for a, ref in [("anti_commit", "trihope_e256"), ("yoked_random", "trihope_e256"),
                   ("shared_frozen_yoked_random", "shared_frozen_trihope"),
                   ("shared_frozen_anti_commit", "shared_frozen_trihope"),
                   ("shared_frozen_trihope", "shared_frozen_never_commit"),
                   ("schedule_anti", "trihope_e256"), ("schedule_random", "trihope_e256"),
                   ("shared_frozen_schedule_anti", "shared_frozen_trihope"),
                   ("shared_frozen_schedule_random", "shared_frozen_trihope"),
                   ("schedule_anti", "anti_commit"), ("schedule_random", "yoked_random"),
                   ("shared_frozen_schedule_anti", "shared_frozen_anti_commit"),
                   ("shared_frozen_schedule_random", "shared_frozen_yoked_random")]:
        if a in arms and ref in arms:
            dd = [res[(a, s)][1] - res[(ref, s)][1] for s in SEEDS]
            print(f"  {a} - {ref}: {' / '.join(f'{x:+.3f}' for x in dd)}  "
                  f"(same sign in {max(sum(x > 0 for x in dd), sum(x < 0 for x in dd))}/3)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

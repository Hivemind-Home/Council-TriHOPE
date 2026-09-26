"""Shared loading, metrics and statistics for the paper's generated numbers.

Everything the paper reports is computed here from ``data/`` (see
``snapshot_data.py``) and nowhere else. Definitions follow ``PREREG.md``:

* M1  = mean over {general, code, math, medical} of that domain's largest
        retention delta over the phase boundaries (analysis/tables.py).
* final = final macro validation loss (mean of the four domains, step 5999).
* permanent writes = routed base-weight coordinate·steps (commit + merges).

Runs are pooled only when their resolved configs are identical except for
seed and output paths (``config_hash``).
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOMAINS = ("general", "code", "math", "medical")
OWN_PHASE = {"code": "code_recurrent", "math": "math_recurrent", "medical": "medical_recurrent"}
PREV_PHASE = {"code": "general_warm", "math": "novel_inject", "medical": "math_recurrent"}
PHASES = (
    ("general_warm", 0, 499), ("code_recurrent", 500, 1999), ("novel_inject", 2000, 2149),
    ("math_recurrent", 2150, 3649), ("medical_recurrent", 3650, 4949),
    ("code_revisit", 4950, 5399), ("mixed_tail", 5400, 5999),
)
DELTA_M1 = 0.02     # PREREG: resolution threshold on M1 (CAMPAIGN_NOTES §6)
DELTA_L = 0.015     # PREREG: resolution threshold on final macro loss
IGNORE_EXACT = {"train.seed", "checkpoint.dir", "logging.path", "logging.events_path", "run.dir"}
IGNORE_PREFIX = ("hydra.",)

# Which spec id names a config-hash group (first match wins). Groups whose
# specs are not listed are named by their first spec id.
NAME_PRIORITY = (
    "trihope", "trihope_sentinel", "trihope_replay", "no_retrieval", "fp_only",
    "trihope_no_hash", "trihope_no_hash_replay",
)


def flatten(d: object, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def config_hash(cfg_path: Path) -> str:
    if not cfg_path.exists():
        return "noconfig"
    flat = flatten(yaml.safe_load(cfg_path.read_text()))
    items = sorted(
        (k, json.dumps(v, sort_keys=True, default=str))
        for k, v in flat.items()
        if k not in IGNORE_EXACT and not k.startswith(IGNORE_PREFIX)
    )
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12]


@dataclass
class Run:
    mode: str
    matrix: str
    spec: str
    seed: int
    path: Path
    summary: dict
    metrics: list[dict]
    config: dict
    chash: str
    m: dict = field(default_factory=dict)  # derived per-run metrics

    @property
    def is_bad_teacher(self) -> bool:
        return self.matrix.startswith("bad_teacher")


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_metrics(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def phase_of_step(step: int) -> str:
    for name, a, b in PHASES:
        if a <= step <= b:
            return name
    return "?"


def derive(run: Run) -> dict:
    s = run.summary
    hist = (s.get("retention") or {}).get("history") or []
    worst: dict[str, float] = {}
    all_deltas: list[float] = []
    for e in hist:
        for d, dl in (e.get("retention_delta") or {}).items():
            worst[d] = max(worst.get(d, float("-inf")), float(dl))
            all_deltas.append(float(dl))
    m: dict = {}
    for d in DOMAINS:
        m[f"worst_{d}"] = worst.get(d)
    vals = [worst[d] for d in DOMAINS if d in worst]
    m["M1"] = mean(vals) if len(vals) == len(DOMAINS) else None
    m["M1max"] = max(vals) if vals else None
    m["meanret"] = mean(all_deltas) if all_deltas else None
    final = (hist[-1].get("loss") or {}) if hist else {}
    for d in DOMAINS:
        m[f"final_{d}"] = _f(final.get(d))
    fl = [m[f"final_{d}"] for d in DOMAINS]
    m["final"] = mean(fl) if all(v is not None for v in fl) else None
    em = (hist[-1].get("em") or {}) if hist else {}
    for d in ("code", "math", "medical"):
        m[f"em_{d}"] = _f(em.get(d))
    # own-phase loss and in-phase learning gain (plasticity)
    by_phase = {e.get("phase"): (e.get("loss") or {}) for e in hist}
    for d, ph in OWN_PHASE.items():
        own = _f(by_phase.get(ph, {}).get(d))
        before = _f(by_phase.get(PREV_PHASE[d], {}).get(d))
        m[f"own_{d}"] = own
        m[f"gain_{d}"] = (before - own) if (own is not None and before is not None) else None
    # anytime macro loss: mean of every eval/macro_loss point
    macro = [_f(r.get("eval/macro_loss")) for r in run.metrics]
    macro = [v for v in macro if v is not None]
    m["anytime"] = mean(macro) if macro else None
    # steps to recover code in code_revisit (5 % of code loss at end of code_recurrent)
    base = _f(by_phase.get("code_recurrent", {}).get("code"))
    rec = None
    if base is not None:
        for r in run.metrics:
            st = int(float(r["step"]))
            v = _f(r.get("eval/code/loss"))
            if v is None or not (4950 <= st <= 5399):
                continue
            if v <= base * 1.05:
                rec = st - 4950
                break
    m["recov"] = rec
    pw = s.get("permanent_writes") or {}
    m["pw"] = _f(pw.get("total")) or 0.0
    m["pw_commit"] = _f(pw.get("p_action_coords")) or 0.0
    m["pw_merge"] = _f(pw.get("merged_coords")) or 0.0
    lt = s.get("ledger_totals") or {}
    for k in ("r_count", "f_count", "p_count", "consolidations", "replay_count", "coords_opened",
              "p_coords_opened"):
        m[k] = _f(lt.get(k))
    m["active"] = _f(s.get("active_fraction_mean"))
    m["replayed_rows"] = _f(((s.get("extra") or {}).get("retrieval") or {}).get("replayed_total"))
    prof = s.get("profile") or {}
    m["wall_s"] = _f(prof.get("wall_clock_s"))
    m["ctrl_frac"] = _f(prof.get("controller_fraction"))
    m["peak_gb"] = _f(prof.get("final_peak_mem_gb"))
    m["steps"] = _f(prof.get("steps"))
    t2 = run.path / "analysis_t2" / "summary.json"  # operator reduction of events.jsonl
    if t2.exists():
        red = json.loads(t2.read_text())
        m["eff_merges"] = _f((red.get("merges") or {}).get("effective"))
        m["eff_merged_coords"] = _f((red.get("merged_coords") or {}).get("effective"))
    kd = [_f(r.get("loss/kd")) for r in run.metrics]
    kd = [v for v in kd if v is not None]
    m["kd_mean"] = mean(kd) if kd else None
    return m


def load_runs(mode: str = "live") -> list[Run]:
    runs: list[Run] = []
    root = DATA / mode
    if not root.exists():
        return runs
    for matrix in sorted(p for p in root.iterdir() if p.is_dir()):
        for rd in sorted(p for p in matrix.iterdir() if p.is_dir() and not p.name.startswith("_")):
            mm = re.match(r"^(.*)-seed(\d+)$", rd.name)
            if not mm or not (rd / "summary.json").exists():
                continue  # rollback / full-restore replays are handled separately
            cfg_path = rd / "config.yaml"
            run = Run(
                mode=mode, matrix=matrix.name, spec=mm.group(1), seed=int(mm.group(2)), path=rd,
                summary=json.loads((rd / "summary.json").read_text()),
                metrics=load_metrics(rd / "metrics.csv"),
                config=flatten(yaml.safe_load(cfg_path.read_text())) if cfg_path.exists() else {},
                chash=config_hash(cfg_path),
            )
            run.m = derive(run)
            runs.append(run)
    return runs


def group_runs(runs: list[Run]) -> dict[str, list[Run]]:
    """Pool runs by resolved-config hash; name each pool by NAME_PRIORITY."""
    by_hash: dict[str, list[Run]] = {}
    for r in runs:
        by_hash.setdefault(r.chash, []).append(r)
    groups: dict[str, list[Run]] = {}
    for rs in by_hash.values():
        specs = {r.spec for r in rs}
        name = next((n for n in NAME_PRIORITY if n in specs), sorted(specs)[0])
        if rs[0].is_bad_teacher:
            name = f"bt_{name}"
        base, k = name, 2
        while name in groups:   # same spec id, different config (e.g. thresholds)
            name, k = f"{base}_v{k}", k + 1
        groups[name] = sorted(rs, key=lambda r: (r.matrix, r.seed))
    return groups


# ----------------------------------------------------------------------------- stats
def vals(rs: list[Run], key: str) -> list[float]:
    return [r.m[key] for r in rs if r.m.get(key) is not None]


def summarize(rs: list[Run], key: str) -> dict:
    v = vals(rs, key)
    if not v:
        return {"n": 0}
    return {
        "n": len(v), "mean": mean(v), "sd": stdev(v) if len(v) > 1 else None,
        "median": median(v), "min": min(v), "max": max(v), "values": v,
    }


def welch_ci(a: list[float], b: list[float], level: float = 0.95) -> tuple:
    """CI for mean(a) - mean(b) (Welch)."""
    if len(a) < 2 or len(b) < 2:
        return (None, None)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        d = mean(a) - mean(b)
        return (d, d)
    df = (va + vb) ** 2 / ((va**2) / (len(a) - 1) + (vb**2) / (len(b) - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    d = mean(a) - mean(b)
    return (d - t * se, d + t * se)


def perm_p(a: list[float], b: list[float], greater: bool = True) -> float | None:
    """Exact one-sided permutation p for mean(a) - mean(b) (> 0 if greater)."""
    if not a or not b:
        return None
    pooled = a + b
    na = len(a)
    obs = mean(a) - mean(b)
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), na):
        s = set(idx)
        ga = [pooled[i] for i in idx]
        gb = [pooled[i] for i in range(len(pooled)) if i not in s]
        d = mean(ga) - mean(gb)
        total += 1
        if (d >= obs - 1e-12) if greater else (d <= obs + 1e-12):
            count += 1
    return count / total


def pair_wins(a: list[float], b: list[float]) -> tuple[int, int]:
    """#pairs with a < b (a better when lower) out of len(a)*len(b)."""
    return sum(1 for x in a for y in b if x < y), len(a) * len(b)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def compare(arm: list[Run], ref: list[Run], key: str, delta: float) -> dict:
    a, b = vals(arm, key), vals(ref, key)
    if not a or not b:
        return {}
    d = mean(a) - mean(b)
    lo, hi = welch_ci(a, b)
    p_gt, p_lt = perm_p(a, b, True), perm_p(a, b, False)
    p_dir = p_gt if d > 0 else p_lt
    thr = delta * (2 if len(a) == 1 else 1)
    resolved = abs(d) >= thr and p_dir is not None and p_dir <= 0.05
    w, tot = pair_wins(b, a)  # ref better (lower) than arm
    return {"d": d, "lo": lo, "hi": hi, "p_gt": p_gt, "p_lt": p_lt, "p_dir": p_dir,
            "resolved": resolved, "ref_wins": w, "pairs": tot, "ratio": (mean(a) / mean(b)) if mean(b) else None}

#!/usr/bin/env python
"""Generate every number and table the paper prints (tables/*.tex).

    python scripts/build_results.py            # draft: pending arms shown as (pending)
    python scripts/build_results.py --final    # camera: pending rows dropped

Outputs
  tables/numbers.tex        \\res{<arm>}{<metric>} lookup + global constants
  tables/tab_main.tex       Table 1: controllers on the clean live stream
  tables/tab_ablation.tex   ablations (Delta vs the trihope reference pool)
  tables/tab_noisy.tex      noisy-teacher stream (E5, live)
  tables/tab_app_*.tex      appendix tables
  data/derived/*.csv        per-run and per-arm values behind every table
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_lib import (  # noqa: E402
    DELTA_L, DELTA_M1, DOMAINS, PHASES, ROOT, compare, group_runs, holm, load_runs, mean,
    pair_wins, perm_p, phase_of_step, summarize, vals, welch_ci,
)

TABLES = ROOT / "tables"
DERIVED = ROOT / "data" / "derived"

# ----------------------------------------------------------------------------- registry
# (arm key, display name, surface) — surface: which parameters an arm can change.
#   S = shared (tied embedding/LM head, norms) trains every step
#   B = routed base blocks (only when a decision opens them)
#   A = adapters
MAIN_ROWS = [
    ("Evidence-gated routing", [
        ("trihope", r"Gated controller (reference pool)", "S+B+A"),
        ("trihope_no_hash", r"Controller, no bucket-id recurrence", "S+B+A"),
        ("no_retrieval", r"Controller, no \defer tier (no replay)", "S+B"),
        ("no_consolidation", r"Controller, never \commit/\consolidate", "S+A"),
        ("surprise_gate", r"Surprise-only gate", "S+A"),
        ("surprise_gate_s4", r"Surprise-only gate, $\theta_S{=}4$", "S+A"),
    ]),
    ("Controls on which updates become permanent", [
        ("random_commit", r"Random \commit, controller's own \defer", "S+B+A"),
        ("random_routing", r"Random relabel of all decisions", "S+B+A"),
        ("p_only", r"Always \commit", "S+B"),
        ("molf_style_a0p5", r"SNR commit rule, $\tau_{\mathrm{SNR}}{=}0.5$", "S+B"),
        ("molf_style_a0p7", r"SNR commit rule, $\tau_{\mathrm{SNR}}{=}0.7$", "S+B"),
        ("molf_style", r"MoLF EPD rule", "S+B+A"),
        ("full_ft", r"Full fine-tuning", "S+B"),
    ]),
    ("Controls by trainable surface", [
        ("frozen_blocks", r"Shared parameters only", "S"),
        ("lora_only", r"LoRA only (plateau trigger never fired)", "A"),
        ("periodic_merge", r"LoRA + merge every 250 steps", "A+B"),
    ]),
]
SURFACE = {k: surf for _t, rows in MAIN_ROWS for k, _n, surf in rows}
DISPLAY = {k: n for _t, rows in MAIN_ROWS for k, n, _s in rows}
SHARED_PARAMS = 155_648_000  # tied embedding/LM head + RMSNorm gains (606.1M - 450.5M routed)

ABLATION_ROWS = [
    ("no_cosine", r"no $\Cbar$ (agreement gate always passes)"),
    ("no_surprise", r"no surprise"),
    ("moments_optimizer", r"optimizer's own (masked) moments"),
    ("topm_all", r"select all blocks (Top-$M$ = all)"),
    ("stability_instant", r"instantaneous $C$ instead of $\Cbar$"),
    ("no_volatility", r"no volatility"),
    ("no_repetition", r"no recurrence signal"),
    ("no_teacher_conf", r"no confidence gate / weighting"),
    ("topm_2", r"Top-$M$ = 2"),
    ("topk_25", r"Top-$K$ = 25\% of ranks"),
    ("topk_100", r"Top-$K$ = 100\% of ranks"),
    ("consolidation_strict", r"strict consolidation (0.7 / 0.6, period 1000)"),
]
PGATE_ROWS = [
    ("pgate_c0p2", 0.2), ("pgate_c0p35", 0.35), ("trihope", 0.5), ("pgate_c0p65", 0.65),
    ("pgate_c0p8", 0.8),
]
NOISY_ROWS = [
    ("bt_trihope", r"Gated controller"),
    ("bt_trihope_nogate", r"Controller, confidence gate off"),
    ("bt_trihope_lowconf", r"Controller, corrupted rows at confidence 0.2"),
    ("bt_trihope_forced", r"Controller, all adapters merged at step 3000"),
    ("bt_molf_snr", r"SNR commit rule, $\tau_{\mathrm{SNR}}{=}0.5$"),
    ("bt_lora_only", r"LoRA only"),
    ("bt_molf_style", r"MoLF EPD rule"),
    ("bt_full_ft", r"Full fine-tuning"),
]
EXPLORATORY_MATRICES = {"yoked_2x2_v1_live", "student17_v1_live", "order2_v1_live"}
EXCLUDED = {"gold_ce"}  # scored on gold targets (different evaluation); see PREREG
SWEEP_NAMES = {
    "p_study": "commit/merge thresholds lowered (0.35)",
    "p_off_control": "same, commit disabled",
    "p_forced_bad": "forced merge at step 2075",
    "p_forced_plausible": "forced merge at step 2350",
    "p_forced_lowconf": "forced merge at step 3850",
    "trihope_r_terminal": r"Controller, \defer without replay",
}


def pretty(key: str) -> str:
    """Paper name for any arm key (sweeps spelled out)."""
    import re as _re
    if key in DISPLAY:
        return DISPLAY[key]
    if key in SWEEP_NAMES:
        return SWEEP_NAMES[key]
    m = _re.match(r"trihope(_no_hash)?_s(\d+)(?:_r0p(\d+))?$", key)
    if m:
        th_r = f", $\\theta_R{{=}}0.{m.group(3)}$" if m.group(3) else ""
        base = r"Controller, no bucket id" if m.group(1) else r"Controller"
        return rf"{base}, $\theta_S{{=}}{m.group(2)}${th_r}"
    m = _re.match(r"surprise_gate_s(\d+)$", key)
    if m:
        return rf"Surprise-only, $\theta_S{{=}}{m.group(1)}$"
    m = _re.match(r"molf_style_a0p(\d+)$", key)
    if m:
        return rf"SNR commit rule, $\tau_{{\mathrm{{SNR}}}}{{=}}0.{m.group(1)}$"
    m = _re.match(r"plateau_trigger(?:_t0p(\d+))?$", key)
    if m:
        return rf"LoRA + plateau merge (tol.\ 0.{m.group(1) or '01'}; never fired)"
    m = _re.match(r"pgate_c0p(\d+)$", key)
    if m:
        return rf"Controller, $\theta_C{{=}}0.{m.group(1)}$"
    for k, lab in ABLATION_ROWS:
        if k == key:
            return rf"Controller, {lab}"
    return r"\arm{" + key.replace("_", r"\_") + "}"


PRIMARY = {"H1": "random_routing", "H1b": "random_commit", "H2": "p_only", "H3": "no_cosine", "H4": "molf_style_a0p5"}
NONINF = ("frozen_blocks", "no_consolidation", "surprise_gate", "random_commit")


# ----------------------------------------------------------------------------- format
def f3(x):
    return "--" if x is None else f"{x:.3f}"


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def sci(x):
    if x is None:
        return "--"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.1f}{{\times}}10^{{{e}}}}}"


def pm(s: dict, fmt=f3):
    if not s or s.get("n", 0) == 0:
        return "--"
    if s.get("sd") is None:
        return fmt(s["mean"])
    return rf"{fmt(s['mean'])}\,{{\scriptsize$\pm${fmt(s['sd'])}}}"


def pval(p):
    if p is None:
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- build
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="drop pending rows")
    args = ap.parse_args()
    TABLES.mkdir(exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)

    runs = load_runs("live")
    # Designed after the pre-declared results were seen (PREREG addendum 4): kept out of every
    # table and test, reported only as exploratory numbers in the appendix.
    explo = [r for r in runs if r.matrix in EXPLORATORY_MATRICES]
    runs = [r for r in runs if r.matrix not in EXPLORATORY_MATRICES]
    groups = group_runs(runs)
    pool = groups["trihope"]

    # ---- per-run CSV (audit trail)
    keys = sorted({k for r in runs for k in r.m})
    with (DERIVED / "runs_live.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "matrix", "spec", "seed", "config_hash"] + keys)
        for g, rs in sorted(groups.items()):
            for r in rs:
                w.writerow([g, r.matrix, r.spec, r.seed, r.chash] + [r.m.get(k) for k in keys])

    res: dict[tuple[str, str], str] = {}

    def put(arm: str, metric: str, value: str) -> None:
        res[(arm, metric)] = value

    arm_rows = []
    for g, rs in groups.items():
        s1, sf = summarize(rs, "M1"), summarize(rs, "final")
        sa, smax = summarize(rs, "anytime"), summarize(rs, "M1max")
        put(g, "n", str(len(rs)))
        put(g, "M1", f3(s1.get("mean")))
        put(g, "M1sd", f3(s1.get("sd")))
        put(g, "M1pm", pm(s1))
        put(g, "M1med", f3(s1.get("median")))
        put(g, "M1max", f3(smax.get("mean")))
        put(g, "final", f3(sf.get("mean")))
        put(g, "finalsd", f3(sf.get("sd")))
        put(g, "finalpm", pm(sf))
        put(g, "anytime", f3(sa.get("mean")))
        put(g, "meanret", f3(summarize(rs, "meanret").get("mean")))
        pwv = vals(rs, "pw")
        put(g, "pw", sci(mean(pwv)) if pwv else "--")
        put(g, "merges", f"{mean(vals(rs, 'consolidations') or [0]):.0f}")
        put(g, "commits", f"{mean(vals(rs, 'p_count') or [0]):.0f}")
        _pc_pool = mean(vals(pool, "p_count") or [0])
        if _pc_pool:
            put(g, "commitratio", f"{mean(vals(rs, 'p_count') or [0]) / _pc_pool:.1f}")
            put(g, "commitratioR", f"{mean(vals(rs, 'p_count') or [0]) / _pc_pool:.0f}")
        put(g, "replayrows", f"{mean(vals(rs, 'replayed_rows') or [0]):.0f}")
        pwc, pwm = mean(vals(rs, "pw_commit") or [0]), mean(vals(rs, "pw_merge") or [0])
        put(g, "pwcommit", sci(pwc))
        put(g, "pwmerge", sci(pwm))
        if pwc + pwm > 0:
            put(g, "mergepct", f"{100 * pwm / (pwc + pwm):.0f}")
        shared_cs = SHARED_PARAMS * (6000 + mean(vals(rs, "replayed_rows") or [0]))
        put(g, "sharedcs", sci(shared_cs))
        if pwv and mean(pwv) > 0:
            put(g, "sharedratio", f"{shared_cs / mean(pwv):.0f}")
        put(g, "active", f"{100 * mean(vals(rs, 'active') or [0]):.3f}")
        dsh = []
        for r in rs:
            m_ = r.m
            if None in (m_.get("p_count"), m_.get("r_count"), m_.get("f_count"), m_.get("replay_count")):
                continue
            top = m_["p_count"] + m_["r_count"] + m_["f_count"] - m_["replay_count"]
            if top > 0:
                dsh.append(100 * m_["r_count"] / top)
        if dsh:
            put(g, "pctdefer", f"{mean(dsh):.1f}")
        put(g, "ctrlfrac", f"{100 * mean(vals(rs, 'ctrl_frac') or [0]):.0f}")
        for d in ("code", "math", "medical"):
            put(g, f"gain{d}", f3(summarize(rs, f"gain_{d}").get("mean")))
            put(g, f"own{d}", f3(summarize(rs, f"own_{d}").get("mean")))
        ov = [own_mean(r) for r in rs if own_mean(r) is not None]
        if ov:
            put(g, "own", f3(mean(ov)))
            gv = [gain_mean(r) for r in rs if gain_mean(r) is not None]
            if gv:
                put(g, "gain", f3(mean(gv)))
            po = [own_mean(r) for r in pool if own_mean(r) is not None]
            if rs is not pool and po:
                w, tot = pair_wins(po, ov)  # pool lower (learned more) than arm
                put(g, "ownwins", f"{w}/{tot}")
                put(g, "pown", pval(perm_p(ov, po, True)))
                put(g, "down", f"{mean(ov) - mean(po):+.3f}")
                put(g, "downabs", f"{abs(mean(ov) - mean(po)):.3f}")
        for d in DOMAINS:
            put(g, f"worst{d}", f3(summarize(rs, f"worst_{d}").get("mean")))
            put(g, f"final{d}", f3(summarize(rs, f"final_{d}").get("mean")))
        rec = [r.m["recov"] for r in rs]
        ok = [x for x in rec if x is not None]
        put(g, "recov", f"{mean(ok):.0f}" if ok else "never")
        put(g, "recovn", f"{len(ok)}/{len(rec)}")
        ref = groups["bt_trihope"] if g.startswith("bt_") and "bt_trihope" in groups else pool
        c1 = compare(rs, ref, "M1", DELTA_M1)
        cf = compare(rs, ref, "final", DELTA_L)
        if c1 and rs is not ref:
            put(g, "ratiomed", f"{summarize(rs, 'M1')['median'] / summarize(ref, 'M1')['mean']:.1f}")
            rr, rp = vals(rs, "replayed_rows"), vals(ref, "replayed_rows")
            if rr and rp and mean(rp) > 0:
                put(g, "replayratio", f"{mean(rr) / mean(rp):.1f}")
            if c1["lo"] is not None:
                put(g, "dMlo", f"{c1['lo']:+.3f}")
                put(g, "dMhi", f"{c1['hi']:+.3f}")
            put(g, "dM1", f"{c1['d']:+.3f}")
            put(g, "dMabs", f"{abs(c1['d']):.3f}")
            put(g, "ratio", f"{c1['ratio']:.1f}")
            put(g, "pM1", pval(c1["p_dir"]))
            put(g, "pM1gt", pval(c1["p_gt"]))
            put(g, "wins", f"{c1['ref_wins']}/{c1['pairs']}")
            put(g, "dM1ci", f"[{f3(c1['lo'])}, {f3(c1['hi'])}]" if c1["lo"] is not None else "--")
            put(g, "resolvedM1", "yes" if c1["resolved"] else "no")
        if cf and rs is not ref:
            put(g, "dfinal", f"{cf['d']:+.3f}")
            put(g, "dfabs", f"{abs(cf['d']):.3f}")
            put(g, "pfinal", pval(cf["p_dir"]))
            put(g, "resolvedfinal", "yes" if cf["resolved"] else "no")
            put(g, "finalwins", f"{cf['ref_wins']}/{cf['pairs']}")
        arm_rows.append((g, len(rs), s1.get("mean"), s1.get("sd"), sf.get("mean"), sf.get("sd"),
                         mean(pwv) if pwv else None, c1.get("d") if c1 else None,
                         c1.get("p_dir") if c1 else None, cf.get("d") if cf else None,
                         cf.get("p_dir") if cf else None))
    with (DERIVED / "arms_live.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n", "M1", "M1_sd", "final", "final_sd", "perm_writes",
                    "dM1_vs_ref", "p_M1", "dfinal_vs_ref", "p_final"])
        w.writerows(arm_rows)

    # ---- primary family (Holm) and non-inferiority
    prim = {h: perm_p(vals(groups[a], "M1"), vals(pool, "M1"), True)
            for h, a in PRIMARY.items() if a in groups}
    for h, p in holm(prim).items():
        put("primary", h, pval(p))
        put("primary", h + "raw", pval(prim[h]))
    for a in NONINF:
        if a in groups:
            lo, hi = welch_ci(vals(pool, "M1"), vals(groups[a], "M1"), level=0.90)
            put(a, "niHi", f3(hi))
            put(a, "niLo", f3(lo))
            put(a, "niEq", f3(max(abs(lo), abs(hi))))
            put(a, "noninferior", "yes" if hi is not None and hi < DELTA_M1 else "no")

    # ---- Theorem 1 at scale: shared-only runs must leave every routed coordinate bit-identical
    wd = [d for d in (read_first_json(r.path / "weight_drift.json") for r in groups.get("frozen_blocks", [])) if d]
    if wd:
        put("const", "bitSeeds", str(len(wd)))
        put("const", "bitRoutedCoords", f"{wd[0]['routed']['coords']:,}".replace(",", "{,}"))
        put("const", "bitRoutedChanged", str(sum(int(d["routed"]["changed_coords"]) for d in wd)))
        put("const", "bitAdaptersNonzero", str(sum(int(d["adapters"]["nonzero_B"]) for d in wd)))
    sg = [d for d in (read_first_json(r.path / "weight_drift.json") for r in groups.get("surprise_gate_s4", [])) if d]
    if sg:
        ch = [int(d["routed"]["changed_coords"]) for d in sg]
        put("const", "sgBitRuns", str(len(sg)))
        put("const", "sgBitClean", str(sum(1 for c in ch if c == 0)))
        put("const", "sgBitChanged", str(max(ch)))
        put("const", "sgBitMaxAbs", f"{max(float(d['routed']['max_abs']) for d in sg):.5f}")

    # ---- Pareto position on the two pre-declared retention/acquisition axes (clean stream, all
    #      configurations incl. single-seed sweeps): forgetting M1 and own-phase loss, both lower=better
    pm_, po_ = mean(vals(pool, "M1")), mean([own_mean(r) for r in pool])
    n_cfg = n_dom = n_domd = 0
    for g, rs in groups.items():
        if g == "trihope" or g.startswith("bt_") or g in EXCLUDED:
            continue
        ov = [own_mean(r) for r in rs if own_mean(r) is not None]
        if not ov:
            continue
        m_, o_ = mean(vals(rs, "M1")), mean(ov)
        n_cfg += 1
        n_domd += int(m_ < pm_ and o_ < po_)   # would beat the method on both axes
        n_dom += int(m_ >= pm_ and o_ >= po_)  # the method is at least as good on both
    put("const", "nConfigs", str(n_cfg))
    put("const", "nDominated", str(n_dom))
    put("const", "nDominating", str(n_domd))

    # ---- exploratory runs (post-protocol; appendix only)
    modal = load_runs("modal")
    pool_commits = mean(vals(pool, "p_count"))
    s17 = [r for r in modal if r.spec == "trihope17"]
    if s17:
        r = s17[0]
        put("const", "sBigM1", f3(r.m["M1"]))
        put("const", "sBigOwn", f3(own_mean(r)))
        put("const", "sBigFinal", f3(r.m["final"]))
        put("const", "sBigCommits", f"{r.m['p_count']:.0f}")
        put("const", "sBigReplay", f"{r.m.get('replayed_rows') or 0:.0f}")
        put("const", "sBigCtrl", f"{100 * (r.m.get('ctrl_frac') or 0):.0f}")
    reruns = sorted((r for r in modal if r.spec == "yoked_full_lo"), key=lambda r: r.seed)
    if reruns:
        put("const", "aHundredN", str(len(reruns)))
        put("const", "aHundredMone", ", ".join(f3(r.m["M1"]) for r in reruns))
        put("const", "aHundredFinal", ", ".join(f3(r.m["final"]) for r in reruns))
    pm1 = vals(pool, "M1")
    put("pool", "M1min", f3(min(pm1)))
    put("pool", "M1max", f3(max(pm1)))
    for spec, tag in (("yoked_blind_lo", "qsLo"), ("yoked_blind_hi", "qsHi")):
        rr = [r for r in explo if r.spec == spec]
        if rr:
            put("const", f"{tag}M1", f3(mean(vals(rr, "M1"))))
            put("const", f"{tag}N", str(len(rr)))
            put("const", f"{tag}CommitRatio", f"{mean(vals(rr, 'p_count')) / pool_commits:.1f}")

    # ---- replay volume vs forgetting across controller arms on the clean stream
    from scipy import stats as _st
    xs, ys = [], []
    for g, rs in groups.items():
        if g.startswith("bt_") or g in EXCLUDED or g in ("lora_only", "full_ft") or g.startswith("plateau"):
            continue
        rr, m1 = vals(rs, "replayed_rows"), vals(rs, "M1")
        if rr and m1:
            xs.append(mean(rr))
            ys.append(mean(m1))
    if len(xs) > 3:
        rho, pr = _st.spearmanr(xs, ys)
        put("const", "replaySpearman", f"{rho:.2f}")
        put("const", "replaySpearmanN", str(len(xs)))
        put("const", "replaySpearmanP", pval(pr))
        pos = [(x, y) for x, y in zip(xs, ys) if x > 0]
        rho2, pr2 = _st.spearmanr([p_[0] for p_ in pos], [p_[1] for p_ in pos])
        put("const", "replayPosSpearman", f"{rho2:.2f}")
        put("const", "replayPosN", str(len(pos)))
        put("const", "replayPosP", pval(pr2))

    # ---- when commits happen (metrics.csv samples write/p_count every 20 steps)
    starts = {name: a for name, a, _b in PHASES}
    early = late = 0.0
    for r in [r for r in pool if r.matrix.startswith("baselines")]:
        for row in r.metrics:
            pc = row.get("write/p_count")
            if pc in (None, "") or float(pc) <= 0:
                continue
            st = int(float(row["step"]))
            ph = phase_of_step(st)
            if st - starts[ph] < 200:
                early += float(pc)
            else:
                late += float(pc)
    if early + late > 0:
        put("const", "commitEarlyPct", f"{100 * early / (early + late):.0f}")
        put("const", "earlyWindowPct", f"{100 * (200 * 6 + 150) / 6000:.0f}")

    # ---- pool / noise constants
    pv = vals(pool, "M1")
    put("pool", "n", str(len(pv)))
    by_seed: dict[int, list[float]] = {}
    for r in pool:
        by_seed.setdefault(r.seed, []).append(r.m["M1"])
    gaps = [max(v) - min(v) for v in by_seed.values() if len(v) > 1]
    put("pool", "maxRerunGap", f3(max(gaps)) if gaps else "--")
    fgaps = []
    fby: dict[int, list[float]] = {}
    for r in pool:
        fby.setdefault(r.seed, []).append(r.m["final"])
    fgaps = [max(v) - min(v) for v in fby.values() if len(v) > 1]
    put("pool", "maxRerunGapFinal", f3(max(fgaps)) if fgaps else "--")
    put("const", "deltaM", f"{DELTA_M1:.2f}")
    put("const", "deltaL", f"{DELTA_L:.3f}")
    put("const", "nLive", str(len(runs)))
    # decision composition of the reference (trihope seed 1337, E1): derived from ledgers
    e1 = [r for r in pool if r.matrix.startswith("baselines") and r.seed == 1337]
    if e1:
        m = e1[0].m
        put("trihope1337", "commits", f"{m['p_count']:.0f}")
        put("trihope1337", "defers", f"{m['r_count']:.0f}")
        put("trihope1337", "fdecisions", f"{m['f_count']:.0f}")
        put("trihope1337", "replaywrites", f"{m['replay_count']:.0f}")
        put("trihope1337", "merges", f"{m['consolidations']:.0f}")
        # top-M decisions = commits + defers + withholds; ledger f_count also counts
        # replay's adapter writes, which are not top-M decisions
        withhold = m["f_count"] - m["replay_count"]
        topm = m["p_count"] + m["r_count"] + withhold
        put("trihope1337", "withholds", f"{withhold:.0f}")
        put("trihope1337", "replayrows", f"{m['replayed_rows']:.0f}")
        put("trihope1337", "topm", f"{topm:.0f}")
        put("trihope1337", "pctwithhold", f"{100 * withhold / topm:.1f}")
        put("trihope1337", "pctdefer", f"{100 * m['r_count'] / topm:.1f}")
        put("trihope1337", "pctcommit", f"{100 * m['p_count'] / topm:.2f}")

    # ---- post-hoc revision analyses (scripts/revision.py; labelled post hoc in the paper)
    import revision
    disp = {g: pretty(g) for g in groups}
    revision.build(groups, pool, put, disp, SURFACE, SHARED_PARAMS, TABLES, excluded=EXCLUDED,
                   primary=PRIMARY)

    write_numbers(res)
    write_main(groups, args.final)
    write_ablation(groups, pool, args.final)
    write_noisy(groups, args.final)
    write_appendix(groups, pool)
    write_cost(groups)
    write_cache(groups)
    holm_txt = ", ".join(f"{h}={res[('primary', h)]}" for h in prim)
    print(f"built: {len(runs)} runs, {len(groups)} groups, pool n={len(pv)}; primary (Holm): {holm_txt}")


def own_mean(r) -> float | None:
    """Acquisition: mean over code/math/medical of each domain's loss at the end of its own
    phase (pre-declared secondary metric "own-phase loss"; lower = learned more in-phase)."""
    v = [r.m.get(f"own_{d}") for d in ("code", "math", "medical")]
    return mean(v) if all(x is not None for x in v) else None


def gain_mean(r) -> float | None:
    v = [r.m.get(f"gain_{d}") for d in ("code", "math", "medical")]
    return mean(v) if all(x is not None for x in v) else None


def read_first_json(path: Path) -> dict | None:
    """First JSON object in a file (some weight_drift.json files carry a stale tail)."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(path.read_text().lstrip())
        return obj
    except (OSError, ValueError):
        return None


def pubkey(arm: str) -> str:
    """Macro key printed into numbers.tex: the code's spec ids start with the retired method
    name; the paper refers to the reference controller as ``ref``."""
    import re as _re
    return _re.sub(r"(^|_)trihope", r"\1ref", arm)


def write_numbers(res: dict) -> None:
    lines = [
        "% GENERATED by scripts/build_results.py -- do not edit by hand.",
        r"\makeatletter",
        r"\newcommand{\res@set}[3]{\expandafter\def\csname res@#1@#2\endcsname{#3}}",
    ]
    for (arm, metric), v in sorted(res.items()):
        lines.append(rf"\res@set{{{pubkey(arm)}}}{{{metric}}}{{{v}}}")
    lines += [
        r"\newcommand{\res}[2]{\ifcsname res@#1@#2\endcsname\csname res@#1@#2\endcsname"
        r"\else\textbf{\textcolor{red}{[#1/#2]}}\fi}",
        r"\makeatother",
    ]
    (TABLES / "numbers.tex").write_text("\n".join(lines) + "\n")


def own_cell(rs: list) -> str:
    ov = [own_mean(r) for r in rs if own_mean(r) is not None]
    if not ov:
        return "--"
    if len(ov) == 1:
        return f3(ov[0])
    from statistics import stdev
    return rf"{mean(ov):.3f}\,{{\scriptsize$\pm${stdev(ov):.3f}}}"


def mark(c: dict | None) -> str:
    """Resolved-difference marker vs the reference (PREREG)."""
    if not c:
        return ""
    if not c["resolved"]:
        return r"$\approx$"
    return r"$\blacktriangle$" if c["d"] > 0 else r"$\triangledown$"


def lg10(x) -> str:
    """log10 of a write count, one decimal; 0 stays 0 (unambiguous under any rendering)."""
    return "0" if not x else f"{math.log10(x):.1f}"


def write_main(groups: dict, final: bool) -> None:
    """Table 1a (arms with three stream seeds, primary) and 1b (fewer seeds, exploratory)."""
    import revision
    pool = groups["trihope"]
    pool_sm = revision.seed_means(pool, "M1")
    head = [
        r"\toprule",
        r" & & & \multicolumn{4}{c}{$\log_{10}$ routed base writes} & & & & & & \\",
        r"\cmidrule(lr){4-7}",
        r"Arm & Surface & runs & commit & merge & perm. & eff. & $\log_{10}$ adapter & Replayed & "
        r"\Mone\ (forgetting) $\downarrow$ & seeds & Own-phase $\downarrow$ & Final loss $\downarrow$ \\",
        r"\midrule",
    ]
    cols = r"\begin{tabular}{@{}l c r r r r r r r l c l l@{}}"
    out_a = ["% GENERATED by scripts/build_results.py -- do not edit by hand.", cols] + head
    out_b = ["% GENERATED by scripts/build_results.py -- do not edit by hand.", cols] + head
    for gi, (title, rows) in enumerate(MAIN_ROWS):
        ra, rb = [], []
        for key, name, surf in rows:
            rs = groups.get(key)
            if not rs:
                continue
            f = revision.arm_features(rs, SHARED_PARAMS, surf)
            s1, sf = summarize(rs, "M1"), summarize(rs, "final")
            c1 = None if key == "trihope" else compare(rs, pool, "M1", DELTA_M1)
            cf = None if key == "trihope" else compare(rs, pool, "final", DELTA_L)
            sm = revision.seed_means(rs, "M1")
            three = set(sm) >= {1337, 2024, 7}
            seeds = "--" if key == "trihope" else (
                f"{sum(sm[x] > pool_sm[x] for x in sm)}/{len(sm)}")
            commit = lg10(f["commit"] + f["unmasked"]) + (r"$^{\ast}$" if f["unmasked"] else "")
            eff = lg10(f["base_effective"])
            name_fmt = rf"\textbf{{{name}}}" if key == "trihope" else name
            line = (rf"\quad {name_fmt} & {surf} & {len(rs)} & {commit} & {lg10(f['merge'])} & "
                    rf"{lg10(f['total'])} & {eff} & {lg10(f['adapter_cs'])} & {f['replay']:.0f} & "
                    rf"{pm(s1)} {mark(c1)} & {seeds} & {own_cell(rs)} & {pm(sf)} {mark(cf)} \\")
            (ra if three else rb).append(line)
        for out, rr in ((out_a, ra), (out_b, rb)):
            if rr:
                if len(out) > len(head) + 2:
                    out.append(r"\addlinespace[2pt]")
                out.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
                out += rr
    # exploratory single-seed ablations that the text discusses
    extra = [("pgate_c0p35", r"Controller, $\theta_C{=}0.35$", "S+B+A"),
             ("topm_all", r"Controller, top-$M$ = all blocks", "S+B+A"),
             ("moments_optimizer", r"Controller, optimizer's own (masked) moments", "S+B+A")]
    rr = []
    for key, name, surf in extra:
        rs = groups.get(key)
        if not rs:
            continue
        f = revision.arm_features(rs, SHARED_PARAMS, surf)
        s1, sf = summarize(rs, "M1"), summarize(rs, "final")
        c1, cf = compare(rs, pool, "M1", DELTA_M1), compare(rs, pool, "final", DELTA_L)
        sm = revision.seed_means(rs, "M1")
        rr.append(rf"\quad {name} & {surf} & {len(rs)} & {lg10(f['commit'])} & {lg10(f['merge'])} & "
                  rf"{lg10(f['total'])} & {lg10(f['base_effective'])} & {lg10(f['adapter_cs'])} & {f['replay']:.0f} & "
                  rf"{pm(s1)} {mark(c1)} & {sum(sm[x] > pool_sm[x] for x in sm)}/{len(sm)} & {own_cell(rs)} & {pm(sf)} {mark(cf)} \\")
    if rr:
        out_b.append(r"\addlinespace[2pt]")
        out_b.append(r"\multicolumn{13}{@{}l}{\emph{Single-seed ablations discussed in the text}} \\")
        out_b += rr
    for out in (out_a, out_b):
        out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_main.tex").write_text("\n".join(out_a) + "\n")
    (TABLES / "tab_main_explo.tex").write_text("\n".join(out_b) + "\n")


def write_cache(live_groups: dict) -> None:
    """Appendix: the text-only (cache-mode) campaign next to the live one."""
    keep = ("baselines_small_v1", "r_tier_small_v1", "bad_teacher_small_v1")
    cache = group_runs([r for r in load_runs("cache") if r.matrix in keep])
    rows = [("trihope", "trihope"), ("no_retrieval", "no_retrieval"),
            ("no_consolidation", "no_consolidation"), ("surprise_gate", "surprise_gate"),
            ("random_routing", "random_routing"), ("lora_only", "lora_only"),
            ("molf_style", "molf_style"), ("p_only", "p_only"),
            ("bt_trihope", "bt_trihope"), ("bt_molf_snr", "bt_molf_snr")]
    out = ["% GENERATED", r"\begin{tabular}{@{}l l l l l@{}}", r"\toprule",
           r"Arm & \Mone\ text-only & \Mone\ live KL & Final text-only & Final live KL \\", r"\midrule"]
    for ck, lk in rows:
        c, l_ = cache.get(ck), live_groups.get(lk)
        if not c or not l_:
            continue
        name = (dict(NOISY_ROWS).get(lk, pretty(lk[3:])) + ", noisy stream") if lk.startswith("bt_") else pretty(lk)
        out.append(rf"{name} & {pm(summarize(c, 'M1'))} & {pm(summarize(l_, 'M1'))} & "
                   rf"{pm(summarize(c, 'final'))} & {pm(summarize(l_, 'final'))} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_app_cache.tex").write_text("\n".join(out) + "\n")


def write_cost(groups: dict) -> None:
    """Appendix: where permanent writes come from, and what is outside the ledger."""
    out = ["% GENERATED", r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
           r"Arm & $\log_{10}$ commit & $\log_{10}$ merge & Eff.\ merges & $\log_{10}$ shared coord$\cdot$steps & Controller time & Wall-clock \\",
           r"\midrule"]
    for _t, rows in MAIN_ROWS:
        for key, name, surf in rows:
            rs = groups.get(key)
            if not rs:
                continue
            commit = mean(vals(rs, "pw_commit") or [0])
            merge = mean(vals(rs, "pw_merge") or [0])
            total = mean(vals(rs, "pw") or [0])
            unmasked = total > 0 and commit == 0 and merge == 0
            rep = mean(vals(rs, "replayed_rows") or [0])
            merges = mean(vals(rs, "consolidations") or [0])
            eff = [r.m.get("eff_merges") for r in rs if r.m.get("eff_merges") is not None]
            if merges == 0:
                eff_txt = "--"
            elif eff:  # measured from the event log (operator task T2)
                eff_txt = f"{mean(eff):.0f} / {merges:.0f}" if merges >= 1 else f"{mean(eff):.1f} / {merges:.1f}"
            elif surf.startswith("A"):  # controller off: adapters train every step
                eff_txt = f"{merges:.0f} / {merges:.0f}"
            elif rep == 0:  # no replay: merged adapters were never written
                eff_txt = f"0 / {merges:.0f}"
            else:
                eff_txt = f"? / {merges:.0f}"
            shared = 0.0 if not surf.startswith("S") else SHARED_PARAMS * (6000 + rep)
            ctrl = mean(vals(rs, "ctrl_frac") or [0]) * 100
            wall = mean(vals(rs, "wall_s") or [0]) / 60
            commit_txt = rf"{lg10(total)}$^{{\ast}}$" if unmasked else lg10(commit)
            out.append(rf"{name} & {commit_txt} & {lg10(merge)} & {eff_txt} & {lg10(shared)} & "
                       rf"{ctrl:.0f}\% & {wall:.0f} min \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_app_cost.tex").write_text("\n".join(out) + "\n")


def write_ablation(groups: dict, pool: list, final: bool) -> None:
    out = [
        "% GENERATED by scripts/build_results.py -- do not edit by hand.",
        r"\begin{tabular}{@{}l r r r r r@{}}",
        r"\toprule",
        r"Change to the controller & $n$ & $\Delta$\Mone & $\Delta$Final & $\log_{10}$ perm.\ writes & Merges \\",
        r"\midrule",
    ]
    for key, name in ABLATION_ROWS:
        rs = groups.get(key)
        if not rs:
            continue
        c1 = compare(rs, pool, "M1", DELTA_M1)
        cf = compare(rs, pool, "final", DELTA_L)
        merges = mean(vals(rs, "consolidations") or [0])
        out.append(
            rf"{name} & {len(rs)} & {c1['d']:+.3f} {mark(c1)} & {cf['d']:+.3f} {mark(cf)} & "
            rf"{lg10(mean(vals(rs, 'pw')))} & {merges:.0f} \\"
        )
    out += [r"\midrule",
            rf"Reference pool & {len(pool)} & \multicolumn{{2}}{{l}}{{\Mone\ {pm(summarize(pool, 'M1'))}}} & "
            rf"{lg10(mean(vals(pool, 'pw')))} & {mean(vals(pool, 'consolidations')):.0f} \\",
            r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_ablation.tex").write_text("\n".join(out) + "\n")

    # C-bar gate dose-response
    out = ["% GENERATED", r"\begin{tabular}{@{}r r l l r@{}}", r"\toprule",
           r"$\theta_C$ (commit) & $n$ & \Mone $\downarrow$ & Final $\downarrow$ & $\log_{10}$ perm.\ writes \\", r"\midrule"]
    for key, c in [("no_consolidation", None)] + PGATE_ROWS + [("no_cosine", None)]:
        rs = groups.get(key)
        label = {"no_consolidation": r"never commit", "no_cosine": r"$\Cbar$ gate off"}.get(key, f"{c}")
        if not rs:
            if not final:
                out.append(rf"{label} & -- & \multicolumn{{3}}{{l}}{{\emph{{running}}}} \\")
            continue
        out.append(rf"{label} & {len(rs)} & {pm(summarize(rs, 'M1'))} & {pm(summarize(rs, 'final'))} & "
                   rf"{lg10(mean(vals(rs, 'pw')))} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_pgate.tex").write_text("\n".join(out) + "\n")


def write_noisy(groups: dict, final: bool) -> None:
    ref = groups.get("bt_trihope")
    if not ref:
        return
    out = ["% GENERATED by scripts/build_results.py -- do not edit by hand.",
           r"\begin{tabular}{@{}l r l l r r@{}}", r"\toprule",
           r"Arm (noisy-teacher stream) & $n$ & \Mone $\downarrow$ & Final $\downarrow$ & $\log_{10}$ perm.\ writes & controller better \\",
           r"\midrule"]
    for key, name in NOISY_ROWS:
        rs = groups.get(key)
        if not rs:
            continue
        s1, sf = summarize(rs, "M1"), summarize(rs, "final")
        wins = "--" if key == "bt_trihope" else "{}/{}".format(*pair_wins(vals(ref, "M1"), vals(rs, "M1")))
        out.append(rf"{name} & {len(rs)} & {pm(s1)} & {pm(sf)} & {lg10(mean(vals(rs, 'pw')))} & {wins} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_noisy.tex").write_text("\n".join(out) + "\n")


def write_appendix(groups: dict, pool: list) -> None:
    # every arm on the clean stream, sorted by M1 (exploratory single seeds included)
    out = ["% GENERATED", r"\begin{tabular}{@{}l r l r l r r@{}}", r"\toprule",
           r"Arm & $n$ & \Mone & median & Final & $\log_{10}$ perm.\ writes & Merges \\", r"\midrule"]
    for g, rs in sorted(groups.items(), key=lambda kv: summarize(kv[1], "M1").get("mean", 9)):
        if g.startswith("bt_") or g in EXCLUDED:
            continue
        s1, sf = summarize(rs, "M1"), summarize(rs, "final")
        name = pretty(g)
        out.append(rf"{name} & {len(rs)} & {pm(s1)} & {f3(s1.get('median'))} & {pm(sf)} & "
                   rf"{lg10(mean(vals(rs, 'pw')))} & {mean(vals(rs, 'consolidations') or [0]):.0f} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_app_allarms.tex").write_text("\n".join(out) + "\n")

    # Delta vs pool with Welch 95% CI (PREREG: effects with CIs)
    out = ["% GENERATED", r"\begin{tabular}{@{}l r r l l@{}}", r"\toprule",
           r"Arm & $n$ & $\Delta$\Mone & 95\% CI & $\Delta$Final (95\% CI) \\", r"\midrule"]
    for _t, rows in MAIN_ROWS:
        for key, name, _s in rows:
            rs = groups.get(key)
            if not rs or key == "trihope" or len(rs) < 2:
                continue
            c1, cf = compare(rs, pool, "M1", DELTA_M1), compare(rs, pool, "final", DELTA_L)
            out.append(rf"{name} & {len(rs)} & {c1['d']:+.3f} & [{c1['lo']:+.3f}, {c1['hi']:+.3f}] & "
                       rf"{cf['d']:+.3f} [{cf['lo']:+.3f}, {cf['hi']:+.3f}] \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_app_ci.tex").write_text("\n".join(out) + "\n")

    # per-domain worst retention deltas and learning gains for the main arms
    out = ["% GENERATED", r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
           r"Arm & \multicolumn{4}{c}{Worst retention $\Delta$ (general / code / math / medical)} & "
           r"\multicolumn{3}{c}{In-phase gain (code / math / medical)} \\", r"\midrule"]
    for _, rows in MAIN_ROWS:
        for key, name, _s in rows:
            rs = groups.get(key)
            if not rs:
                continue
            w = [f3(summarize(rs, f"worst_{d}").get("mean")) for d in DOMAINS]
            gn = [f3(summarize(rs, f"gain_{d}").get("mean")) for d in ("code", "math", "medical")]
            out.append(rf"{name} & {' & '.join(w)} & {' & '.join(gn)} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_app_domains.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

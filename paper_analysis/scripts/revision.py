"""Post-hoc analyses added in the pre-submission revision (all labelled post hoc in the paper).

Called from ``build_results.py`` after the pre-declared analyses, so every number here is
regenerated from ``data/`` by the same ``make results`` step. Nothing here replaces a
pre-declared test; each block reports *alongside* them.

W1  seed-level statistics: variance decomposition of M1 in the reference pool, seed-level
    means, paired per-seed differences, the five confirmatory tests with the stream seed as
    the exchangeable unit, minimum attainable p and minimum detectable effect.
W2  write reconciliation: authorized vs effective vs realized writes, arithmetic audit of
    Perm. writes = Commit + Merge, realized drift from ``weight_drift.json``.
W3  what predicts forgetting across configurations: OLS of M1 on log write volumes by
    surface (base commits, effective merges, adapter writes, replay rows), with VIFs,
    partial R^2 and leave-one-out residuals; diagnosis of the two counterexample arms.
W4  the computable part of Proposition 2' (commit term) next to realized drift.
Opt metric robustness: alternative forgetting summaries and their rank agreement with M1.

Quantities that cannot be recovered from the snapshot (per-merge delta norms, l1 drift,
per-row validation losses, drift for runs without ``weight_drift.json``) are reported as
missing, never estimated.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

from results_lib import DATA, DELTA_L, DELTA_M1, DOMAINS, vals

SEEDS = (1337, 2024, 7)
ETA, WD = 1e-4, 0.01          # learning rate and weight decay of every arm (app:hparams)
GAMMA_INF = 7.27              # Prop. app-step, beta = (0.9, 0.999)
LORA_SCALE = 32 / 16          # alpha / r


# ----------------------------------------------------------------------------- helpers
def read_first_json(path: Path) -> dict | None:
    try:
        obj, _ = json.JSONDecoder().raw_decode(path.read_text().lstrip())
        return obj
    except (OSError, ValueError):
        return None


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def sgn3(x):
    return "--" if x is None else f"{x:+.3f}"


def pv(p):
    return "--" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def sci_tex(x):
    if not x:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.1f}{{\times}}10^{{{e}}}}}"


def lg(x):
    """log10 write count, printed with one decimal (0 stays 0)."""
    if x is None:
        return "--"
    return "0" if x <= 0 else f"{math.log10(x):.1f}"


def seed_means(rs: list, key: str) -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for r in rs:
        v = r.m.get(key)
        if v is not None:
            by.setdefault(r.seed, []).append(v)
    return {s: mean(v) for s, v in by.items()}


def signflip_p(d: list[float]) -> float:
    """Exact one-sided sign-flip p for mean(d) > 0 (paired, seed = unit)."""
    obs = mean(d)
    cnt = tot = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        tot += 1
        cnt += mean([s * x for s, x in zip(signs, d)]) >= obs - 1e-12
    return cnt / tot


def perm_p_unpaired(a: list[float], b: list[float]) -> float:
    pooled = a + b
    obs = mean(a) - mean(b)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        s = set(idx)
        ga = [pooled[i] for i in idx]
        gb = [pooled[i] for i in range(len(pooled)) if i not in s]
        tot += 1
        cnt += (mean(ga) - mean(gb)) >= obs - 1e-12
    return cnt / tot


def t_ci(d: list[float], level=0.95):
    if len(d) < 2:
        return (None, None)
    se = stdev(d) / math.sqrt(len(d))
    t = stats.t.ppf(0.5 + level / 2, len(d) - 1)
    return (mean(d) - t * se, mean(d) + t * se)


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


# ----------------------------------------------------------------------------- features
def adapter_cs(m: dict) -> float:
    co, pc = m.get("coords_opened") or 0.0, m.get("p_coords_opened") or 0.0
    return max(0.0, co - pc - (m.get("unmasked_base") or 0.0))


def realized(r) -> dict | None:
    d = read_first_json(r.path / "weight_drift.json")
    if not d:
        return None
    return {"routed_l2": float(d["routed"]["l2_delta"]),
            "routed_changed": int(d["routed"]["changed_coords"]),
            "routed_max": float(d["routed"]["max_abs"]),
            "routed_coords": int(d["routed"]["coords"]),
            "shared_l2": float(d["shared"]["l2_delta"]),
            "shared_changed": int(d["shared"]["changed_coords"]),
            "adapter_frob": float(d["adapters"]["sum_frob_BA"]),
            "adapter_nonzeroB": int(d["adapters"]["nonzero_B"])}


def arm_features(rs: list, shared_params: int, surface: str | None) -> dict:
    def avg(k):
        v = [r.m.get(k) for r in rs if r.m.get(k) is not None]
        return mean(v) if v else None
    for r in rs:  # unmasked base coordinate.steps (full FT) from the summary
        r.m.setdefault("unmasked_base", float((r.summary.get("permanent_writes") or {})
                                                .get("unmasked_base_coords_total") or 0))
    rep = avg("replayed_rows") or 0.0
    merges = avg("consolidations") or 0.0
    eff = avg("eff_merged_coords")
    if eff is None and merges == 0:
        eff = 0.0
    trains_shared = surface is None or surface.startswith("S")
    rz = [x for x in (realized(r) for r in rs) if x]
    f = {
        "n": len(rs),
        "commit": avg("pw_commit") or 0.0,
        "merge": avg("pw_merge") or 0.0,
        "unmasked": avg("unmasked_base") or 0.0,
        "total": avg("pw") or 0.0,
        "merges": merges,
        "eff_merges": avg("eff_merges") if merges else 0.0,
        "eff_merge_cs": eff,
        "adapter_cs": mean([adapter_cs(r.m) for r in rs]),
        "replay": rep,
        "shared_cs": shared_params * (6000 + rep) if trains_shared else 0.0,
        "n_drift": len(rz),
        "routed_l2": mean([x["routed_l2"] for x in rz]) if rz else None,
        "shared_l2": mean([x["shared_l2"] for x in rz]) if rz else None,
        "M1": avg("M1"), "final": avg("final"),
    }
    f["base_effective"] = f["commit"] + (f["eff_merge_cs"] or 0.0) + f["unmasked"]
    f["vol"] = f["base_effective"] + f["adapter_cs"]   # every non-shared write that can change the function
    return f


# ----------------------------------------------------------------------------- main entry
def build(groups: dict, pool: list, put, display, surface_of, shared_params, tables: Path,
          excluded=frozenset(), primary: dict | None = None) -> dict:
    out: dict = {}
    derived = DATA / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    def disp(g):
        return display.get(g) or g.replace("_", r"\_")

    clean = {g: rs for g, rs in groups.items() if not g.startswith("bt_") and g not in excluded}
    for rs in groups.values():          # acquisition (own-phase loss) per run, as in build_results.own_mean
        for r in rs:
            v = [r.m.get(f"own_{d}") for d in ("code", "math", "medical")]
            r.m["own"] = mean(v) if all(x is not None for x in v) else None
    full3 = {g: rs for g, rs in clean.items() if set(seed_means(rs, "M1")) >= set(SEEDS)
             and not g.startswith("plateau")}

    # ===================================================================== W1
    # --- variance decomposition of M1 (and final loss) in the reference pool
    for key, tag in (("M1", "M"), ("final", "L")):
        by: dict[int, list[float]] = {}
        for r in pool:
            by.setdefault(r.seed, []).append(r.m[key])
        k = len(by)
        N = sum(len(v) for v in by.values())
        grand = mean([x for v in by.values() for x in v])
        ssw = sum((x - mean(v)) ** 2 for v in by.values() for x in v)
        ssb = sum(len(v) * (mean(v) - grand) ** 2 for v in by.values())
        msw, msb = ssw / (N - k), ssb / (k - 1)
        n0 = (N - sum(len(v) ** 2 for v in by.values()) / N) / (k - 1)
        s2_stream = max(0.0, (msb - msw) / n0)
        icc = s2_stream / (s2_stream + msw) if (s2_stream + msw) > 0 else 0.0
        F = msb / msw if msw > 0 else float("inf")
        pF = 1 - stats.f.cdf(F, k - 1, N - k)
        out[f"var_{tag}"] = dict(s2_stream=s2_stream, s2_gpu=msw, icc=icc, F=F, pF=pF,
                                 n0=n0, k=k, N=N, by={s: v for s, v in by.items()})
        put("rev", f"sdStream{tag}", f3(math.sqrt(s2_stream)))
        put("rev", f"sdGpu{tag}", f3(math.sqrt(msw)))
        put("rev", f"icc{tag}", f"{icc:.2f}")
        put("rev", f"F{tag}", f"{F:.1f}")
        put("rev", f"pF{tag}", pv(pF))
        put("rev", f"varStreamPct{tag}", f"{100 * icc:.0f}")
    vM = out["var_M"]
    # seed-level sd of a single seed-mean (what one seed of a new arm is compared against)
    sd_seedmean = math.sqrt(vM["s2_stream"] + vM["s2_gpu"])  # one run per seed
    put("rev", "sdSeedRun", f3(sd_seedmean))

    # --- seed-level paired differences, all arms with all three seeds
    pool_sm = {k: seed_means(pool, k) for k in ("M1", "final", "own")}
    rows = []
    for g, rs in sorted(full3.items(), key=lambda kv: mean(vals(kv[1], "M1"))):
        if g == "trihope":
            continue
        rec = {"g": g, "n": len(rs)}
        for key in ("M1", "final", "own"):
            sm = seed_means(rs, key)
            if set(sm) < set(SEEDS):
                continue
            d = [sm[s] - pool_sm[key][s] for s in SEEDS]
            lo, hi = t_ci(d)
            lo90, hi90 = t_ci(d, 0.90)
            rec[key] = dict(d=d, mean=mean(d), sd=stdev(d), lo=lo, hi=hi, lo90=lo90, hi90=hi90,
                            k_worse=sum(x > 0 for x in d), p_pair=signflip_p(d),
                            p_pair_lt=signflip_p([-x for x in d]),
                            p_unp=perm_p_unpaired([sm[s] for s in SEEDS],
                                                  [pool_sm[key][s] for s in SEEDS]),
                            seed_arm=[sm[s] for s in SEEDS])
        rows.append(rec)
        out.setdefault("seed", {})[g] = rec
        put(g, "sdM", sgn3(rec["M1"]["mean"]))
        put(g, "sdMlo", sgn3(rec["M1"]["lo"]))
        put(g, "sdMhi", sgn3(rec["M1"]["hi"]))
        put(g, "sdMk", f"{rec['M1']['k_worse']}/3")
        put(g, "sdMkBetter", f"{3 - rec['M1']['k_worse']}/3")
        put(g, "sdMp", pv(rec["M1"]["p_pair"]))
        put(g, "sdMloN", sgn3(rec["M1"]["lo90"]))
        put(g, "sdMhiN", sgn3(rec["M1"]["hi90"]))
        put(g, "sdMeqN", f3(max(abs(rec["M1"]["lo90"]), abs(rec["M1"]["hi90"]))))
        put(g, "sdMequiv", "yes" if max(abs(rec["M1"]["lo90"]), abs(rec["M1"]["hi90"])) < DELTA_M1 else "no")
        put(g, "sdF", sgn3(rec["final"]["mean"]))
        put(g, "sdFlo", sgn3(rec["final"]["lo"]))
        put(g, "sdFhi", sgn3(rec["final"]["hi"]))
        put(g, "sdFk", f"{rec['final']['k_worse']}/3")
        if "own" in rec:
            put(g, "sdO", sgn3(rec["own"]["mean"]))
            put(g, "sdOlo", sgn3(rec["own"]["lo"]))
            put(g, "sdOhi", sgn3(rec["own"]["hi"]))
            put(g, "sdOk", f"{rec['own']['k_worse']}/3")
        put(g, "sdFkBetter", f"{3 - rec['final']['k_worse']}/3")

    # pooled sd of paired seed differences across arms (for the paired MDE)
    ssd = [r["M1"]["sd"] ** 2 for r in rows]
    sd_d = math.sqrt(mean(ssd)) if ssd else None
    lvl = [r["M1"]["sd"] ** 2 for r in rows if abs(r["M1"]["mean"]) < DELTA_M1]
    sd_d_level = math.sqrt(mean(lvl)) if lvl else None
    ssdF = [r["final"]["sd"] ** 2 for r in rows]
    sd_dF = math.sqrt(mean(ssdF)) if ssdF else None
    lvlF = [r["final"]["sd"] ** 2 for r in rows if abs(r["final"]["mean"]) < DELTA_L]
    sd_dF_level = math.sqrt(mean(lvlF)) if lvlF else None
    tp = stats.t.ppf(0.95, 2) + stats.t.ppf(0.80, 2)          # paired, 3 seeds, one-sided
    tu = stats.t.ppf(0.95, 4) + stats.t.ppf(0.80, 4)          # unpaired, 3 vs 3 seed means
    mde_pair = tp * sd_d_level / math.sqrt(3) if sd_d_level else None
    mde_unp = tu * sd_seedmean * math.sqrt(2 / 3)
    mde_pairF = tp * sd_dF_level / math.sqrt(3) if sd_dF_level else None
    sdF_seed = math.sqrt(out["var_L"]["s2_stream"] + out["var_L"]["s2_gpu"])
    mde_unpF = tu * sdF_seed * math.sqrt(2 / 3)
    out["mde"] = dict(pair=mde_pair, unp=mde_unp, pairF=mde_pairF, unpF=mde_unpF,
                      sd_d=sd_d, sd_d_level=sd_d_level)
    put("rev", "sdPairDiff", f3(sd_d_level))
    put("rev", "sdPairDiffAll", f3(sd_d))
    put("rev", "mdePair", f3(mde_pair))
    put("rev", "mdeUnp", f3(mde_unp))
    put("rev", "mdePairF", f3(mde_pairF))
    put("rev", "mdeUnpF", f3(mde_unpF))
    put("rev", "mdePairRatio", f"{mde_pair / DELTA_M1:.1f}" if mde_pair else "--")
    put("rev", "mdePairFRatio", f"{mde_pairF / DELTA_L:.1f}" if mde_pairF else "--")
    put("rev", "nSeedArms", str(len(rows)))
    put("rev", "minPpair", f"{1 / 8:.3f}")
    put("rev", "minPunp", f"{1 / 20:.3f}")
    put("rev", "minHolmPair", f"{min(1.0, 5 / 8):.3f}")
    put("rev", "minHolmUnp", f"{min(1.0, 5 / 20):.3f}")

    # --- the five confirmatory tests, seed as the unit (reported alongside the pre-declared ones)
    conf_rows = []
    if primary:
        praw_pair, praw_unp = {}, {}
        for h, a in primary.items():
            rs = groups.get(a)
            if not rs:
                continue
            sm = seed_means(rs, "M1")
            seeds = [s for s in SEEDS if s in sm]
            d = [sm[s] - pool_sm["M1"][s] for s in seeds]
            praw_pair[h] = signflip_p(d)
            praw_unp[h] = perm_p_unpaired([sm[s] for s in seeds], [pool_sm["M1"][s] for s in SEEDS])
            conf_rows.append((h, a, len(rs), seeds, d))
        hp, hu = holm(praw_pair), holm(praw_unp)
        for h, a, n, seeds, d in conf_rows:
            put("rev", f"{h}pair", pv(praw_pair[h]))
            put("rev", f"{h}pairHolm", pv(hp[h]))
            put("rev", f"{h}unp", pv(praw_unp[h]))
            put("rev", f"{h}unpHolm", pv(hu[h]))
            put("rev", f"{h}k", f"{sum(x > 0 for x in d)}/{len(d)}")
        out["confirm"] = dict(rows=conf_rows, pair=praw_pair, unp=praw_unp, hp=hp, hu=hu)

    # --- pooling vs seed convention for the arms the text discusses
    # (the run-pool "k/30 pairings" treat same-seed reruns as independent; reported as such)

    # ===================================================================== W2
    feats = {g: arm_features(rs, shared_params, surface_of.get(g)) for g, rs in clean.items()}
    out["feats"] = feats
    audit_bad = []
    for g, rs in clean.items():
        for r in rs:
            pw = r.summary.get("permanent_writes") or {}
            tot = float(pw.get("total") or 0)
            parts = float(pw.get("p_action_coords") or 0) + float(pw.get("merged_coords") or 0) \
                + float(pw.get("unmasked_base_coords_total") or 0)
            if abs(tot - parts) > 0.5:
                audit_bad.append((g, r.seed, tot, parts))
            lt = r.summary.get("ledger_totals") or {}
            if abs(float(lt.get("p_coords_opened") or 0) - float(pw.get("p_action_coords") or 0)) > 0.5:
                audit_bad.append((g, r.seed, "ledger_vs_pw_commit"))
            if abs(float(lt.get("merged_coords") or 0) - float(pw.get("merged_coords") or 0)) > 0.5:
                audit_bad.append((g, r.seed, "ledger_vs_pw_merge"))
    n_audit = sum(len(rs) for rs in clean.values())
    put("rev", "auditRuns", str(n_audit))
    put("rev", "auditBad", str(len(audit_bad)))
    out["audit_bad"] = audit_bad
    # effective share of the reference budget
    fp = feats["trihope"]
    put("rev", "refEffMergePct", f"{100 * fp['eff_merge_cs'] / fp['merge']:.0f}" if fp["merge"] else "--")
    put("rev", "refIneffPct", f"{100 * (fp['merge'] - fp['eff_merge_cs']) / fp['total']:.0f}")
    put("rev", "refEffTotal", f"{fp['base_effective']:.2g}")
    put("rev", "refEffLog", lg(fp["base_effective"]))
    put("rev", "refAdapterLog", lg(fp["adapter_cs"]))
    for g in ("trihope", "random_commit", "no_retrieval", "no_cosine", "p_only", "molf_style", "full_ft",
              "periodic_merge", "topm_all", "moments_optimizer", "lora_only", "frozen_blocks",
              "molf_style_a0p7", "pgate_c0p35", "no_consolidation", "surprise_gate_s4"):
        if g not in feats:
            continue
        f = feats[g]
        for k in ("commit", "merge", "total", "eff_merge_cs", "base_effective", "adapter_cs", "shared_cs"):
            put(g, f"log{k.replace('_', '')}", lg(f[k]))
        put(g, "effmerges", f"{f['eff_merges']:.0f}" if f["merges"] else "0")
        put(g, "adapterRatio", f"{f['adapter_cs'] / fp['adapter_cs']:.0f}" if fp["adapter_cs"] else "--")
        put(g, "commitWriteRatio", f"{f['commit'] / fp['commit']:.0f}" if fp["commit"] else "--")
        put(g, "totalWriteRatio", f"{f['total'] / fp['total']:.0f}" if fp["total"] else "--")
        put(g, "effWriteRatio", f"{f['base_effective'] / fp['base_effective']:.1f}"
            if fp["base_effective"] else "--")
        if f["routed_l2"] is not None:
            put(g, "routedLtwo", f"{f['routed_l2']:.2f}")
            put(g, "sharedLtwo", f"{f['shared_l2']:.0f}")
            put(g, "ndrift", str(f["n_drift"]))
    # random commit vs reference: total and effective writes
    rc = feats.get("random_commit")
    if rc:
        put("rev", "rcTotalExcessPct", f"{100 * (rc['total'] / fp['total'] - 1):.0f}")
        put("rev", "rcEffExcessPct", f"{100 * (rc['base_effective'] / fp['base_effective'] - 1):.0f}")
        put("rev", "rcCommitRatio", f"{rc['commit'] / fp['commit']:.2f}")
        put("rev", "rcMergeRatio", f"{rc['merge'] / fp['merge']:.2f}")
        put("rev", "rcEffMergeRatio", f"{rc['eff_merge_cs'] / fp['eff_merge_cs']:.2f}")
        put("rev", "rcReplayRatio", f"{rc['replay'] / fp['replay']:.2f}")
    # realized drift coverage and correlation with M1 across runs that have it
    drift_runs = []
    for g, rs in clean.items():
        for r in rs:
            z = realized(r)
            if z:
                drift_runs.append((g, r, z))
    put("rev", "nDriftRuns", str(len(drift_runs)))
    zw = [(g, r, z) for g, r, z in drift_runs if feats[g]["total"] == 0 and feats[g]["unmasked"] == 0]
    put("rev", "zeroWriteRuns", str(len(zw)))
    put("rev", "zeroWriteBad", str(sum(z["routed_changed"] > 0 for g, r, z in zw)))
    put("rev", "zeroWriteChanged", str(sum(z["routed_changed"] for g, r, z in zw)))
    checks = sum(z["routed_coords"] for g, r, z in zw)
    put("rev", "zeroWriteChecks", sci_tex(checks))
    put("rev", "zeroWriteArms", str(len({g for g, r, z in zw})))
    put("rev", "zeroWriteReplayRuns", str(sum(feats[g]["replay"] > 0 for g, r, z in zw)))
    put("rev", "nDriftArms", str(len({g for g, *_ in drift_runs})))
    pool_drift = [z for g, r, z in drift_runs if g == "trihope"]
    if pool_drift:
        put("rev", "poolDriftN", str(len(pool_drift)))
        put("rev", "poolRoutedLtwo", f"{mean(z['routed_l2'] for z in pool_drift):.2f}")
        put("rev", "poolSharedLtwo", f"{mean(z['shared_l2'] for z in pool_drift):.0f}")
        put("rev", "sharedOverRouted", f"{mean(z['shared_l2'] for z in pool_drift) / mean(z['routed_l2'] for z in pool_drift):.0f}")
    xs = [z["routed_l2"] for g, r, z in drift_runs]
    ys = [r.m["M1"] for g, r, z in drift_runs]
    if len(xs) > 3:
        rho, p = stats.spearmanr(xs, ys)
        put("rev", "driftSpearman", f"{rho:.2f}")
        put("rev", "driftSpearmanP", pv(p))
        # without the two untouched-base arms (drift = 0) and periodic merge (different surface)
        sub = [(x, y) for (g, r, z), x, y in zip(drift_runs, xs, ys)
               if g not in ("frozen_blocks", "surprise_gate_s4", "periodic_merge")]
        rho2, p2 = stats.spearmanr([a for a, _ in sub], [b for _, b in sub])
        put("rev", "driftSpearmanCtl", f"{rho2:.2f}")
        put("rev", "driftSpearmanCtlP", pv(p2))
        put("rev", "driftSpearmanCtlN", str(len(sub)))
    out["drift_runs"] = drift_runs

    # Prop 2' commit term, computable per arm: eta * n * (Gamma + lambda*Theta) summed over
    # commit coordinate.steps; compared in l2 with realized routed drift via Cauchy-Schwarz
    # (||d||_2 <= ||d||_1). Theta = 1 bounds |theta| for these weights? not logged -> use the
    # l_inf of the pretrained routed weights is not in the snapshot, so we report the bound
    # for lambda*Theta <= 0.01 * 1 (|theta| <= 1 for Qwen3-0.6B projections is NOT verified):
    # we print it as a function of Theta, and only the Theta-free leading term as a number.
    slacks = []
    for g in ("trihope", "random_commit", "no_cosine", "molf_style_a0p7", "pgate_c0p35"):
        f = feats.get(g)
        if not f:
            continue
        lead = ETA * GAMMA_INF * f["commit"]      # Theta-free part of the commit term (l1)
        rz = [realized(r) for r in clean[g]]
        rz = [z for z in rz if z]
        if not rz:
            continue
        # ||d||_1 <= sqrt(#changed) * ||d||_2  (Cauchy-Schwarz over the changed coordinates)
        l1_up = mean(math.sqrt(z["routed_changed"]) * z["routed_l2"] for z in rz)
        put(g, "boundLead", sci_tex(lead))
        put(g, "realLoneUp", sci_tex(l1_up))
        put(g, "boundSlack", f"{lead / l1_up:.0f}")
        slacks.append(lead / l1_up)
    if slacks:
        put("rev", "boundSlackMin", f"{min(slacks):.0f}")
        put("rev", "boundSlackMax", f"{max(slacks):.0f}")
        put("rev", "boundNarms", str(len(slacks)))

    # ===================================================================== W3
    ok = [g for g, f in feats.items() if f["M1"] is not None and not g.startswith("plateau")]

    def design(gs, cols):
        X = np.array([[math.log10(feats[g][c] + 1) for c in cols] for g in gs])
        y = np.array([feats[g]["M1"] for g in gs])
        return X, y

    cols = ["commit", "eff_merge_cs", "adapter_cs", "replay"]
    labels = {"commit": r"$\log_{10}$(commit coord$\cdot$steps$+1$)",
              "eff_merge_cs": r"$\log_{10}$(effective merge coord$\cdot$steps$+1$)",
              "adapter_cs": r"$\log_{10}$(adapter coord$\cdot$steps$+1$)",
              "replay": r"$\log_{10}$(replayed rows$+1$)",
              "unmasked": r"$\log_{10}$(unmasked base coord$\cdot$steps$+1$)"}
    regs = {}
    subsets = {
        "n3": [g for g in ok if feats[g]["n"] >= 3 and g != "full_ft"],
        "all": [g for g in ok if g != "full_ft"],
        "allft": ok,
    }
    for name, gs in subsets.items():
        cc = cols + (["unmasked"] if name == "allft" else [])
        X, y = design(gs, cc)
        regs[name] = ols(X, y, cc)
        regs[name]["gs"] = gs
    out["regs"] = regs
    r3 = regs["n3"]
    put("rev", "regNthree", str(len(r3["gs"])))
    put("rev", "regNall", str(len(regs["all"]["gs"])))
    put("rev", "regRtwoThree", f"{r3['r2']:.2f}")
    put("rev", "regRtwoAll", f"{regs['all']['r2']:.2f}")
    for name in ("n3", "all"):
        rr = regs[name]
        for c in cols:
            j = rr["cols"].index(c)
            t = name.replace("n3", "Three").replace("all", "All")
            cname = c.replace("_", "")
            put("rev", f"b{cname}{t}", f"{rr['b'][j + 1]:+.3f}")
            put("rev", f"pr{cname}{t}", f"{rr['partial_r2'][j]:.2f}")
            put("rev", f"p{cname}{t}", pv(rr["p"][j + 1]))
            put("rev", f"vif{cname}{t}", f"{rr['vif'][j]:.1f}")
    # single-predictor rank correlations across all clean configurations
    sp = {}
    for c in cols + ["commit_plus_merge", "total", "base_effective", "vol"]:
        xs_ = [feats[g]["commit"] + feats[g]["merge"] if c == "commit_plus_merge" else feats[g][c] for g in subsets["all"]]
        ys_ = [feats[g]["M1"] for g in subsets["all"]]
        rho, p = stats.spearmanr(xs_, ys_)
        sp[c] = (rho, p)
        put("rev", f"rho{c.replace('_', '')}", f"{rho:.2f}")
        put("rev", f"rhoP{c.replace('_', '')}", pv(p))
    out["spearman"] = sp
    # within the gated range: arms whose authorized routed writes are within 10x of the reference
    ref_t = feats["trihope"]["total"]
    gr = [g for g in subsets["all"] if 0 < feats[g]["total"] <= 10 * ref_t and g != "trihope"] + ["trihope"]
    for c in ("total", "base_effective", "vol"):
        rho, p = stats.spearmanr([feats[g][c] for g in gr], [feats[g]["M1"] for g in gr])
        put("rev", f"rhoGated{c.replace('_', '')}", f"{rho:.2f}")
        put("rev", f"rhoPGated{c.replace('_', '')}", pv(p))
    put("rev", "nGated", str(len(gr)))
    m1g = [feats[g]["M1"] for g in gr]
    put("rev", "gatedMlo", f3(min(m1g)))
    put("rev", "gatedMhi", f3(max(m1g)))
    for name, gs in (("Three", subsets["n3"]), ("All", subsets["all"])):
        for c in ("total", "base_effective", "adapter_cs", "replay", "vol", "commit"):
            rho, p = stats.spearmanr([feats[g][c] for g in gs], [feats[g]["M1"] for g in gs])
            put("rev", f"rho{c.replace('_', '')}{name}", f"{rho:.2f}")
            put("rev", f"rhoP{c.replace('_', '')}{name}", pv(p))
        put("rev", f"rhoN{name}", str(len(gs)))
    # leave-one-out residuals of the two counterexamples in the all-config fit
    for g in ("topm_all", "moments_optimizer"):
        if g in regs["all"]["gs"]:
            j = regs["all"]["gs"].index(g)
            put(g, "resid", f"{regs['all']['resid'][j]:+.3f}")
            put(g, "residLoo", f"{regs['all']['loo'][j]:+.3f}")
    # routed-only model (commit + effective merge) residuals for the counterexamples
    Xr, yr = design(subsets["all"], ["commit", "eff_merge_cs"])
    rr = ols(Xr, yr, ["commit", "eff_merge_cs"])
    out["reg_routed"] = rr
    put("rev", "regRtwoRouted", f"{rr['r2']:.2f}")
    for g in ("topm_all", "moments_optimizer"):
        if g in subsets["all"]:
            put(g, "residRouted", f"{rr['resid'][subsets['all'].index(g)]:+.3f}")

    # --- counterexample diagnosis from the decision logs
    for g in ("topm_all", "moments_optimizer", "trihope"):
        rs = clean.get(g) or []
        dsum = {"defer": 0, "withhold": 0, "commit": 0, "adapt": 0, "tot": 0}
        for r in rs:
            p = r.path / "analysis_t2" / "decisions.csv"
            if not p.exists():
                continue
            for row in csv.DictReader(p.open()):
                c = int(row["count"])
                dsum["tot"] += c
                if row["action"] == "R":
                    dsum["defer"] += c
                elif row["action"] == "P":
                    dsum["commit"] += c
                elif row["surface"] == "adapter":
                    dsum["adapt"] += c
                else:
                    dsum["withhold"] += c
        if dsum["tot"]:
            for k in ("defer", "withhold", "commit", "adapt"):
                put(g, f"share{k}", f"{100 * dsum[k] / dsum['tot']:.1f}")
        f = feats.get(g)
        if f:
            put(g, "replayRatioRef", f"{f['replay'] / fp['replay']:.1f}")
            put(g, "replayrowsRev", f"{f['replay']:.0f}")
            put(g, "decisionsPerStep", f"{dsum['tot'] / 6000 / max(1, len(rs)):.0f}" if dsum["tot"] else "--")

    import re as _re
    sweep_pat = _re.compile(r"^(trihope(_no_hash)?_s\d|surprise_gate_s\d|molf_style_a0p\d|pgate_c0p\d|plateau_trigger)")
    cfg = [g for g in clean if g != "trihope" and not g.startswith("bt_") and clean[g]
           and all(r.m.get("own") is not None for r in clean[g])]
    put("rev", "nCfg", str(len(cfg)))
    put("rev", "nCfgThree", str(sum(set(seed_means(clean[g], "M1")) >= set(SEEDS) for g in cfg)))
    put("rev", "nCfgSingle", str(sum(len(clean[g]) == 1 for g in cfg)))
    put("rev", "nCfgSweep", str(sum(bool(sweep_pat.match(g)) and len(clean[g]) == 1 for g in cfg)))

    # ===================================================================== optional: metric robustness
    alt = {}
    for g, rs in clean.items():
        vv = {"M1": [], "M1max": [], "meanret": [], "avgforget": [], "bwt": []}
        for r in rs:
            hist = (r.summary.get("retention") or {}).get("history") or []
            if not hist or r.m.get("M1") is None:
                continue
            vv["M1"].append(r.m["M1"])
            vv["M1max"].append(r.m["M1max"])
            vv["meanret"].append(r.m["meanret"])
            # standard average forgetting: per domain, max over later boundaries of (best loss
            # reached at the end of any training phase of d so far) -> loss now increase; we use
            # the final boundary before the mixed tail (code_revisit end) as "now".
            fin = next((e for e in hist if e.get("phase") == "code_revisit"), None)
            best = {}
            for e in hist:
                ph = e.get("phase")
                for d, ph_own in (("general", "general_warm"), ("code", "code_recurrent"),
                                  ("math", "math_recurrent"), ("medical", "medical_recurrent")):
                    if ph == ph_own or (d == "code" and ph == "code_revisit"):
                        L = (e.get("loss") or {}).get(d)
                        if L is not None:
                            best[d] = min(best.get(d, float("inf")), float(L))
            if fin:
                fl = fin.get("loss") or {}
                af = [float(fl[d]) - best[d] for d in DOMAINS if d in best and d in fl]
                vv["avgforget"].append(mean(af) if af else None)
                # BWT (loss form): mean over domains trained before the revisit of
                # loss at end of code_revisit minus loss at end of own phase (positive = forgot)
                own = {e.get("phase"): (e.get("loss") or {}) for e in hist}
                bw = []
                for d, ph_own in (("general", "general_warm"), ("code", "code_recurrent"),
                                  ("math", "math_recurrent"), ("medical", "medical_recurrent")):
                    if d in fl and d in own.get(ph_own, {}):
                        bw.append(float(fl[d]) - float(own[ph_own][d]))
                vv["bwt"].append(mean(bw) if bw else None)
        alt[g] = {k: mean([x for x in v if x is not None]) for k, v in vv.items() if any(x is not None for x in v)}
    out["alt"] = alt
    ks = [g for g in alt if all(k in alt[g] for k in ("M1", "M1max", "meanret", "avgforget", "bwt"))]
    for k in ("M1max", "meanret", "avgforget", "bwt"):
        rho, _ = stats.spearmanr([alt[g]["M1"] for g in ks], [alt[g][k] for g in ks])
        put("rev", f"rank{k}", f"{rho:.2f}")
        k3 = [g for g in ks if len(clean[g]) >= 3]
        rho3, _ = stats.spearmanr([alt[g]["M1"] for g in k3], [alt[g][k] for g in k3])
        put("rev", f"rankThree{k}", f"{rho3:.2f}")
    put("rev", "rankN", str(len(ks)))
    put("rev", "rankNthree", str(len([g for g in ks if len(clean[g]) >= 3])))
    # does any n>=3 arm change sign relative to the pool when the metric changes?
    flips = []
    for g in ks:
        if g == "trihope" or len(clean[g]) < 3:
            continue
        s0 = np.sign(alt[g]["M1"] - alt["trihope"]["M1"])
        for k in ("M1max", "meanret", "avgforget", "bwt"):
            if abs(alt[g]["M1"] - alt["trihope"]["M1"]) >= DELTA_M1 and \
                    np.sign(alt[g][k] - alt["trihope"][k]) != s0:
                flips.append((g, k))
    put("rev", "metricFlips", str(len(flips)))
    out["flips"] = flips

    # ===================================================================== tables
    write_tables(out, groups, pool, disp, tables, feats, full3)
    write_csvs(out, derived, feats)
    return out


def ols(X: np.ndarray, y: np.ndarray, cols: list[str]) -> dict:
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ b
    resid = y - yhat
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    dof = n - k - 1
    s2 = sse / dof if dof > 0 else float("nan")
    try:
        cov = s2 * np.linalg.inv(Xc.T @ Xc)
        se = np.sqrt(np.diag(cov))
        tt = b / se
        p = 2 * (1 - stats.t.cdf(np.abs(tt), dof))
    except np.linalg.LinAlgError:
        se = p = np.full(k + 1, np.nan)
    # partial R^2 of each predictor (drop-one)
    pr2, vif = [], []
    for j in range(k):
        keep = [i for i in range(k) if i != j]
        Xr = np.column_stack([np.ones(n), X[:, keep]])
        br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        sser = float(((y - Xr @ br) ** 2).sum())
        pr2.append((sser - sse) / sser if sser > 0 else float("nan"))
        bj, *_ = np.linalg.lstsq(Xr, X[:, j], rcond=None)
        rj = X[:, j] - Xr @ bj
        r2j = 1 - float(rj @ rj) / float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        vif.append(1 / (1 - r2j) if r2j < 1 else float("inf"))
    # leave-one-out residuals
    H = Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T
    loo = resid / np.clip(1 - np.diag(H), 1e-9, None)
    return dict(b=b, se=se, p=p, r2=r2, resid=resid, loo=loo, yhat=yhat, partial_r2=pr2, vif=vif,
                cols=cols, n=n, dof=dof)


# ----------------------------------------------------------------------------- tables
def write_tables(out, groups, pool, disp, T: Path, feats, full3):
    # --- variance decomposition
    vM, vL = out["var_M"], out["var_L"]
    L = ["% GENERATED by scripts/revision.py (post hoc) -- do not edit by hand.",
         r"\begin{tabular}{@{}l r r r r l@{}}", r"\toprule",
         r"Metric & $\sigma_{\text{stream}}$ & $\sigma_{\text{GPU}}$ & share stream & $F_{2,7}$ & per-seed pool runs (mean) \\",
         r"\midrule"]
    for name, v in (("\\Mone", vM), ("Final loss", vL)):
        per = "; ".join(f"{s}: {len(x)} ({mean(x):.3f})" for s, x in sorted(v["by"].items()))
        L.append(rf"{name} & {math.sqrt(v['s2_stream']):.4f} & {math.sqrt(v['s2_gpu']):.4f} & "
                 rf"{100 * v['icc']:.0f}\% & {v['F']:.1f} ($p={pv(v['pF'])}$) & {per} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (T / "tab_rev_variance.tex").write_text("\n".join(L) + "\n")

    # --- seed-level paired differences
    L = ["% GENERATED by scripts/revision.py (post hoc)",
         r"\begin{tabular}{@{}l r l r l r l r@{}}", r"\toprule",
         r"Arm & runs & \multicolumn{3}{c}{$\Delta$\Mone vs.\ reference, per seed} & "
         r"\multicolumn{3}{c}{$\Delta$Final vs.\ reference, per seed} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r" & & mean [95\% CI] & worse & 1337 / 2024 / 7 & mean & [95\% CI] & worse \\", r"\midrule"]
    for g, rec in sorted(out.get("seed", {}).items(), key=lambda kv: kv[1]["M1"]["mean"]):
        a, b = rec["M1"], rec["final"]
        per = " / ".join(f"{x:+.3f}" for x in a["d"])
        L.append(rf"{disp(g)} & {rec['n']} & {a['mean']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] & "
                 rf"{a['k_worse']}/3 & {per} & {b['mean']:+.3f} & [{b['lo']:+.3f}, {b['hi']:+.3f}] & {b['k_worse']}/3 \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (T / "tab_rev_seed.tex").write_text("\n".join(L) + "\n")

    # --- confirmatory tests: run-level (pre-declared) next to seed-level (post hoc)
    c = out.get("confirm")
    if c:
        L = ["% GENERATED by scripts/revision.py (post hoc)",
             r"\begin{tabular}{@{}l l r l l l l@{}}", r"\toprule",
             r"Test & Arm & runs & pre-declared, run unit (Holm) & seed unit, paired (Holm) & seed unit, unpaired (Holm) & seeds worse \\",
             r"\midrule"]
        for h, a, n, seeds, d in c["rows"]:
            L.append(rf"{h} & {disp(a)} & {n} & \res{{primary}}{{{h}}} & {pv(c['pair'][h])} ({pv(c['hp'][h])}) & "
                     rf"{pv(c['unp'][h])} ({pv(c['hu'][h])}) & {sum(x > 0 for x in d)}/{len(d)} \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (T / "tab_rev_confirm.tex").write_text("\n".join(L) + "\n")

    # --- write reconciliation
    order = ["trihope", "no_retrieval", "random_commit", "random_routing", "no_cosine", "pgate_c0p35",
             "molf_style_a0p7", "p_only", "molf_style", "full_ft", "no_consolidation", "surprise_gate_s4",
             "frozen_blocks", "lora_only", "periodic_merge", "topm_all", "moments_optimizer"]
    L = ["% GENERATED by scripts/revision.py (post hoc). log10 of coordinate.steps; 0 = none.",
         r"\begin{tabular}{@{}l r r r r r r r r r r r@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{Routed base writes ($\log_{10}$ coord$\cdot$steps)} & "
         r"\multicolumn{2}{c}{Merges} & \multicolumn{2}{c}{Other surfaces} & \multicolumn{2}{c}{Realized $\ell_2$ drift} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
         r"Arm & $n$ & commit & merge & total & effective & eff./all & replayed & adapter & shared & routed & shared \\",
         r"\midrule"]
    for g in order:
        f = feats.get(g)
        if not f:
            continue
        eff = f"{f['eff_merges']:.0f}/{f['merges']:.0f}" if f["merges"] >= 1 else ("--" if not f["merges"] else f"{f['eff_merges']:.1f}/{f['merges']:.1f}")
        commit = lg(f["commit"] + f["unmasked"]) + (r"$^{\ast}$" if f["unmasked"] else "")
        rl2 = "--" if f["routed_l2"] is None else f"{f['routed_l2']:.2f}"
        sl2 = "--" if f["shared_l2"] is None else f"{f['shared_l2']:.0f}"
        nd = "" if f["routed_l2"] is None or f["n_drift"] == f["n"] else rf"$^{{({f['n_drift']})}}$"
        L.append(rf"{disp(g)} & {f['n']} & {commit} & {lg(f['merge'])} & {lg(f['total'])} & "
                 rf"{lg(f['base_effective'])} & {eff} & {f['replay']:.0f} & {lg(f['adapter_cs'])} & "
                 rf"{lg(f['shared_cs'])} & {rl2}{nd} & {sl2} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (T / "tab_rev_writes.tex").write_text("\n".join(L) + "\n")

    # --- regression
    labels = {"commit": r"commit coord$\cdot$steps", "eff_merge_cs": r"effective merge coord$\cdot$steps",
              "adapter_cs": r"adapter coord$\cdot$steps", "replay": r"replayed rows"}
    L = ["% GENERATED by scripts/revision.py (post hoc)",
         r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
         rf" & \multicolumn{{4}}{{c}}{{arms with $n\ge3$ ({out['regs']['n3']['n']} arms, $R^2={out['regs']['n3']['r2']:.2f}$)}} & "
         rf"\multicolumn{{4}}{{c}}{{all configurations ({out['regs']['all']['n']}, $R^2={out['regs']['all']['r2']:.2f}$)}} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"$\log_{10}(\cdot+1)$ of & $b$ & $p$ & partial $R^2$ & VIF & $b$ & $p$ & partial $R^2$ & VIF \\", r"\midrule"]
    for c_ in ("commit", "eff_merge_cs", "adapter_cs", "replay"):
        cells = []
        for name in ("n3", "all"):
            rr = out["regs"][name]
            j = rr["cols"].index(c_)
            cells.append(f"{rr['b'][j + 1]:+.3f} & {pv(rr['p'][j + 1])} & {rr['partial_r2'][j]:.2f} & {rr['vif'][j]:.1f}")
        L.append(rf"{labels[c_]} & {' & '.join(cells)} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (T / "tab_rev_regress.tex").write_text("\n".join(L) + "\n")


def write_csvs(out, derived: Path, feats):
    with (derived / "writes_reconciliation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        keys = ["n", "commit", "merge", "unmasked", "total", "merges", "eff_merges", "eff_merge_cs",
                "base_effective", "adapter_cs", "replay", "shared_cs", "n_drift", "routed_l2", "shared_l2",
                "M1", "final"]
        w.writerow(["arm"] + keys)
        for g, f in sorted(feats.items()):
            w.writerow([g] + [f.get(k) for k in keys])
    with (derived / "seed_level.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "metric", "d_1337", "d_2024", "d_7", "mean", "lo95", "hi95", "p_signflip"])
        for g, rec in sorted(out.get("seed", {}).items()):
            for k in ("M1", "final"):
                a = rec[k]
                w.writerow([g, k, *a["d"], a["mean"], a["lo"], a["hi"], a["p_pair"]])
    with (derived / "drift_runs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "matrix", "seed", "M1", "routed_l2", "routed_changed", "routed_max_abs",
                    "shared_l2", "adapter_frob"])
        for g, r, z in out["drift_runs"]:
            w.writerow([g, r.matrix, r.seed, r.m["M1"], z["routed_l2"], z["routed_changed"],
                        z["routed_max"], z["shared_l2"], z["adapter_frob"]])

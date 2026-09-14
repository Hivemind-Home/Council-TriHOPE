"""Paper-table builders. All functions take a list[RunData] (seeds are
aggregated as mean±std over runs sharing a spec_id) and return DataFrames."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .loaders import RunData


def _agg(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Mean±std over seeds; keeps single-seed values as-is with std NaN."""
    out = df.groupby(group_cols, as_index=False)[value_cols].agg(["mean", "std"])
    out.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in out.columns
    ]
    return out.reset_index() if group_cols[0] not in out.columns else out


def action_share_by_phase(runs: list[RunData]) -> pd.DataFrame:
    """Paper Table 2 analogue: R/F/P action shares + coords opened per phase."""
    rows = []
    for run in runs:
        if run.events.empty:
            continue
        decisions = run.events[run.events["type"] == "decision"].copy()
        if decisions.empty or "action" not in decisions.columns:
            continue
        decisions["phase"] = decisions["step"].map(run.phase_of_step)
        for phase, group in decisions.groupby("phase"):
            total = len(group)
            rows.append(
                {
                    "spec_id": run.spec_id,
                    "seed": run.seed,
                    "phase": phase,
                    "r_share": (group["action"] == "R").mean() * 100,
                    "f_share": (group["action"] == "F").mean() * 100,
                    "p_share": (group["action"] == "P").mean() * 100,
                    "decisions": total,
                    "mean_coords_opened": group["coords_opened"].mean(),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _agg(
        df, ["spec_id", "phase"], ["r_share", "f_share", "p_share", "mean_coords_opened"]
    )


def forgetting_table(runs: list[RunData]) -> pd.DataFrame:
    """Paper Table 1 analogue: final per-domain loss/EM, retention delta,
    peak memory, wall-clock."""
    rows = []
    for run in runs:
        rec: dict = {"spec_id": run.spec_id, "seed": run.seed}
        retention = (run.summary.get("retention") or {}).get("history") or []
        if retention:
            final = retention[-1]
            for d, loss in (final.get("loss") or {}).items():
                rec[f"final_loss_{d}"] = loss
            for d, em in (final.get("em") or {}).items():
                rec[f"final_em_{d}"] = em
            # Worst retention delta seen anywhere in the run = headline forgetting.
            worst: dict[str, float] = {}
            for entry in retention:
                for d, delta in (entry.get("retention_delta") or {}).items():
                    worst[d] = max(worst.get(d, float("-inf")), delta)
            for d, delta in worst.items():
                rec[f"worst_retention_delta_{d}"] = delta
        profile = run.summary.get("profile") or {}
        rec["wall_clock_s"] = profile.get("wall_clock_s")
        rec["tokens_per_sec"] = profile.get("tokens_per_sec")
        rec["peak_mem_gb"] = profile.get("final_peak_mem_gb")
        rec["controller_fraction"] = profile.get("controller_fraction")
        ledger = run.summary.get("ledger_totals") or {}
        for key in ("r_count", "f_count", "p_count", "consolidations"):
            rec[key] = ledger.get(key)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    value_cols = [c for c in df.columns if c not in ("spec_id", "seed")]
    return _agg(df, ["spec_id"], value_cols)


def ablation_deltas(runs: list[RunData], baseline: str = "trihope") -> pd.DataFrame:
    """Per-spec deltas vs the baseline spec on the forgetting-table metrics."""
    table = forgetting_table(runs)
    if table.empty:
        return pd.DataFrame()
    # Pooled reports prefix spec ids with their experiment name
    # (``baselines_small_v1/trihope``); accept that form so the ablation
    # grid can be reported against E1's trihope in one call.
    ids = list(table["spec_id"])
    matches = [s for s in ids if s == baseline or str(s).endswith(f"/{baseline}")]
    if not matches:
        return pd.DataFrame()
    baseline = matches[0]
    base = table[table["spec_id"] == baseline].iloc[0]
    mean_cols = [c for c in table.columns if c.endswith("_mean")]
    rows = []
    for _, row in table.iterrows():
        if row["spec_id"] == baseline:
            continue
        rec = {"spec_id": row["spec_id"]}
        for col in mean_cols:
            if pd.notna(row[col]) and pd.notna(base[col]):
                rec[f"delta_{col.removesuffix('_mean')}"] = row[col] - base[col]
        rows.append(rec)
    return pd.DataFrame(rows)


def _phase_lookup(run: RunData):
    """step -> phase name, from the run's phase_start events."""
    starts = run.phases
    if not starts:
        return lambda step: None
    steps = [s for s, _ in starts]
    names = [n for _, n in starts]

    def lookup(step: int):
        i = int(np.searchsorted(steps, step, side="right")) - 1
        return names[i] if i >= 0 else None

    return lookup


def replay_timing(runs: list[RunData]) -> pd.DataFrame:
    """Where and when deferred rows came back (the R tier's write-back).

    One row per (spec, seed): replay count, the delay between parking
    (``origin_step``) and replay (``step``) in steps, the share of replays
    whose origin lies in an *earlier* phase than the replay (cross-phase =
    stale write-back), and the count per replay phase. Written to separate
    the two explanations for replay's forgetting cost — a late write on top
    of what was learned in between, versus simply more adapter updates —
    which the forgetting table alone cannot tell apart.
    """
    rows = []
    for run in runs:
        ev = run.events
        if ev.empty or "type" not in ev.columns:
            continue
        rep = ev[ev["type"] == "replay"]
        if rep.empty or "origin_step" not in rep.columns:
            continue
        rep = rep.dropna(subset=["origin_step", "step"])
        # One replay = one row; the loop logs one event per opened module,
        # so collapse to (step, origin_step, bucket) first.
        key_cols = [c for c in ("step", "origin_step", "bucket_id") if c in rep.columns]
        rep = rep.drop_duplicates(subset=key_cols)
        phase_of = _phase_lookup(run)
        replay_phase = rep["step"].astype(int).map(phase_of)
        origin_phase = rep["origin_step"].astype(int).map(phase_of)
        delay = rep["step"].astype(int) - rep["origin_step"].astype(int)
        cross = (replay_phase != origin_phase)
        rec: dict = {
            "spec_id": run.spec_id,
            "seed": run.seed,
            "replays": int(len(rep)),
            "delay_median_steps": float(delay.median()),
            "delay_p90_steps": float(delay.quantile(0.9)),
            "cross_phase_share": float(cross.mean()),
            "cross_phase_replays": int(cross.sum()),
        }
        for phase, n in replay_phase.value_counts().items():
            rec[f"replays_in_{phase}"] = int(n)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna(0)


def p_selection_stats(runs: list[RunData]) -> pd.DataFrame:
    """When does P fire, and what do the signals look like at those moments?"""
    rows = []
    for run in runs:
        if run.events.empty or "action" not in run.events.columns:
            continue  # controller-off runs emit no decisions
        p_events = run.events[
            (run.events["type"] == "decision") & (run.events["action"] == "P")
        ]
        merges = run.events[run.events["type"] == "consolidation"]
        rec = {
            "spec_id": run.spec_id,
            "seed": run.seed,
            "p_decisions": len(p_events),
            "consolidations": len(merges),
            "forced_consolidations": (
                int(merges["forced"].sum()) if "forced" in merges and len(merges) else 0
            ),
            "first_p_step": int(p_events["step"].min()) if len(p_events) else None,
            "first_merge_step": int(merges["step"].min()) if len(merges) else None,
        }
        if len(p_events):
            sig = pd.DataFrame(list(p_events["signals"]))
            for col in ("S", "C", "V", "R"):
                if col in sig:
                    rec[f"mean_{col}_at_P"] = float(sig[col].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def _loss_curve(run: RunData, phase: str) -> Optional[pd.DataFrame]:
    if run.metrics.empty or "phase" not in run.metrics.columns:
        return None
    df = run.metrics[
        (run.metrics.get("phase") == phase) & run.metrics["loss/total"].notna()
    ]
    return df[["step", "loss/total"]] if len(df) else None


def adapter_reuse_aulc(
    runs: list[RunData], phase: str = "math_recurrent"
) -> pd.DataFrame:
    """Area-under-training-loss-curve on ``phase`` per spec — the freed-adapter
    speed-of-learning comparison (P-on vs P-off after a consolidation).

    The default is the first recurrent phase *after* the code phases, i.e.
    the first one that can benefit from adapters those consolidations
    freed. It is stream-dependent: pass ``phase`` explicitly if the phase
    order in configs/stream_*.yaml changes.
    """
    rows = []
    for run in runs:
        curve = _loss_curve(run, phase)
        if curve is None or len(curve) < 2:
            continue
        steps = curve["step"].to_numpy(dtype=float)
        losses = curve["loss/total"].to_numpy(dtype=float)
        aulc = float(np.trapezoid(losses, steps) / (steps[-1] - steps[0]))
        rows.append(
            {
                "spec_id": run.spec_id,
                "seed": run.seed,
                "phase": phase,
                "aulc": aulc,
                "final_loss": float(losses[-1]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return _agg(pd.DataFrame(rows), ["spec_id", "phase"], ["aulc", "final_loss"])


def _run_config(run: RunData) -> dict:
    if run.events.empty or "type" not in run.events.columns:
        return {}
    rows = run.events[run.events["type"] == "run_config"]
    if rows.empty:
        return {}
    return {k: v for k, v in rows.iloc[0].to_dict().items() if pd.notna(v) or isinstance(v, dict)}


def threshold_tag(run: RunData) -> str:
    """Compact operating-point label from the ``run_config`` event.

    Read from the trace, not parsed out of the spec id, so a sweep manifest
    can name its runs however it likes. Empty when no trace exists.
    """
    cfg = _run_config(run)
    if not cfg:
        return ""
    mode = str(cfg.get("policy_mode") or "rfp")
    override = cfg.get("policy_override")
    thr = cfg.get("thresholds") or {}
    cons = cfg.get("consolidation") or {}
    parts = [mode if not override else f"{mode}:{override}"]
    if mode == "adam_score":
        parts.append(f"A{cfg.get('adam_score_high')}")
    elif mode == "surprise_only":
        parts.append(f"S{thr.get('surprise_high')}")
    elif mode == "rfp":
        parts.append(f"S{thr.get('surprise_high')}_R{thr.get('repetition_low')}")
    if str(cons.get("trigger", "signals")) == "plateau":
        parts.append(f"plateau{cons.get('plateau_tolerance')}")
    return "_".join(str(p) for p in parts)


def steps_to_recover(
    run: RunData,
    domain: str = "code",
    phase: str = "code_revisit",
    own_phase: str = "code_recurrent",
    tolerance: float = 0.05,
) -> Optional[int]:
    """Steps into ``phase`` until ``eval/{domain}/loss`` is back within
    ``tolerance`` (relative) of its value at the end of ``own_phase``.

    ``None`` when the run never recovers inside the phase, or when the
    baseline / eval rows are missing. Resolution is the eval interval in
    effect during the phase (``eval.interval_by_phase``).
    """
    history = (run.summary.get("retention") or {}).get("history") or []
    baseline = None
    for entry in history:
        if entry.get("phase") == own_phase:
            baseline = (entry.get("loss") or {}).get(domain)
    if baseline is None or run.metrics.empty:
        return None
    col = f"eval/{domain}/loss"
    if col not in run.metrics.columns:
        return None
    start = None
    for s, name in run.phases:
        if name == phase:
            start = s
    if start is None:
        return None
    df = run.metrics[run.metrics[col].notna()].sort_values("step")
    if "phase" in df.columns:
        in_phase = df[df["step"].map(run.phase_of_step) == phase]
    else:
        in_phase = df[df["step"] >= start]
    bar = float(baseline) * (1.0 + tolerance)
    hit = in_phase[in_phase[col] <= bar]
    if hit.empty:
        return None
    return int(hit.iloc[0]["step"]) - int(start)


def budget_curve(
    runs: list[RunData],
    recover_domain: str = "code",
    recover_phase: str = "code_revisit",
    recover_own_phase: str = "code_recurrent",
) -> pd.DataFrame:
    """Figure 1's table: forgetting vs. permanent writes, one row per
    (spec, operating point), mean±std over seeds.

    ``permanent_writes`` counts (coordinate, step) write events on base
    weights — P actions on base params, LoRA→base merges (the merged block's
    base numel), and, for unmasked runs, every trainable base coordinate
    every step — as written into ``run_summary.json["permanent_writes"]``.
    """
    rows = []
    for run in runs:
        pw = run.summary.get("permanent_writes") or {}
        rec: dict = {
            "spec_id": run.spec_id,
            "seed": run.seed,
            "threshold_tag": threshold_tag(run),
            "permanent_writes": pw.get("total"),
            "p_action_coords": pw.get("p_action_coords"),
            "merged_coords": pw.get("merged_coords"),
            "active_fraction_mean": run.summary.get("active_fraction_mean"),
            "replayed_total": ((run.summary.get("extra") or {}).get("retrieval") or {}).get(
                "replayed_total"
            ),
        }
        history = (run.summary.get("retention") or {}).get("history") or []
        deltas: list[float] = []
        worst: Optional[float] = None
        for entry in history:
            for _d, delta in (entry.get("retention_delta") or {}).items():
                deltas.append(float(delta))
                worst = delta if worst is None else max(worst, delta)
        rec["worst_retention_delta"] = worst
        rec["mean_retention_delta"] = float(np.mean(deltas)) if deltas else None
        if history:
            for d, loss in (history[-1].get("loss") or {}).items():
                rec[f"final_loss_{d}"] = loss
        rec["steps_to_recover_code"] = steps_to_recover(
            run, recover_domain, recover_phase, recover_own_phase
        )
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    value_cols = [c for c in df.columns if c not in ("spec_id", "seed", "threshold_tag")]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return _agg(df, ["spec_id", "threshold_tag"], value_cols)


def _corrupted_tag(run: RunData) -> Optional[str]:
    cfg = _run_config(run)
    block = cfg.get("corrupt_teacher") if cfg else None
    if isinstance(block, dict) and block.get("enabled"):
        return str(block.get("tag"))
    return None


def teacher_attribution(runs: list[RunData]) -> pd.DataFrame:
    """Who wrote how much where (task T5.1).

    One row per (run, seed, teacher, phase): R/F/P action counts, replayed
    F writes, coordinates opened in adapters (F) and directly in base
    weights (P on a P-type module), and the consolidations attributed to
    the teacher — each merge's ``attribution_share`` (the share of the
    merged adapter's evidence that teacher wrote, carried on the
    ``consolidation`` event by the ledger) summed over merges, plus the
    base coordinates that share represents.
    """
    rows = []
    for run in runs:
        if run.events.empty or "type" not in run.events.columns:
            continue
        ev = run.events
        acc: dict[tuple[str, str], dict[str, float]] = {}

        def _get(teacher: str, phase: str) -> dict[str, float]:
            key = (teacher, phase)
            if key not in acc:
                acc[key] = {
                    "R_count": 0, "F_count": 0, "P_count": 0, "replay_count": 0,
                    "coords_opened_F": 0, "coords_opened_P": 0,
                    "consolidations_attributed": 0.0, "merged_coords_attributed": 0.0,
                }
            return acc[key]

        if "action" in ev.columns:
            decisions = ev[ev["type"] == "decision"]
            for _, d in decisions.iterrows():
                teacher = str(d.get("teacher") or "")
                phase = str(run.phase_of_step(int(d["step"])) or "")
                rec = _get(teacher, phase)
                action = str(d.get("action"))
                rec[f"{action}_count"] = rec.get(f"{action}_count", 0) + 1
                coords = int(d.get("coords_opened") or 0)
                module = str(d.get("module") or "")
                if action == "P" and module.endswith(".P"):
                    rec["coords_opened_P"] += coords
                elif coords:
                    rec["coords_opened_F"] += coords
            replays = ev[ev["type"] == "replay"]
            for _, d in replays.iterrows():
                teacher = str(d.get("teacher") or "")
                phase = str(run.phase_of_step(int(d["step"])) or "")
                rec = _get(teacher, phase)
                rec["replay_count"] += 1
                rec["coords_opened_F"] += int(d.get("coords_opened") or 0)
        merges = ev[ev["type"] == "consolidation"]
        if "attribution_share" in merges.columns:
            for _, m in merges.iterrows():
                shares = m.get("attribution_share")
                if not isinstance(shares, dict):
                    continue
                phase = str(run.phase_of_step(int(m["step"])) or "")
                merged = float(m.get("merged_coords") or 0.0)
                for teacher, share in shares.items():
                    rec = _get(str(teacher), phase)
                    rec["consolidations_attributed"] += float(share)
                    rec["merged_coords_attributed"] += merged * float(share)
        for (teacher, phase), rec in acc.items():
            rows.append({"run": run.run_id, "spec_id": run.spec_id, "seed": run.seed,
                         "teacher": teacher, "phase": phase, **rec})
    return pd.DataFrame(rows)


def containment(runs: list[RunData], teacher: Optional[str] = None) -> pd.DataFrame:
    """Where did the corrupted teacher's influence end up? (task T5.1)

    Per run: the share of that teacher's actions routed R / F / P, the base
    coordinates it wrote directly, and the consolidations / base
    coordinates attributed to it — then mean±std over seeds per spec. The
    teacher defaults to the run's ``corrupt_teacher.tag``; runs without one
    are skipped.
    """
    attr = teacher_attribution(runs)
    if attr.empty:
        return pd.DataFrame()
    rows = []
    for run in runs:
        tag = teacher or _corrupted_tag(run)
        if not tag:
            continue
        sub = attr[(attr["run"] == run.run_id) & (attr["teacher"] == tag)]
        counts = sub[["R_count", "F_count", "P_count"]].sum() if len(sub) else None
        total = float(counts.sum()) if counts is not None else 0.0
        rows.append(
            {
                "spec_id": run.spec_id,
                "seed": run.seed,
                "teacher": tag,
                "actions": total,
                "r_share": (counts["R_count"] / total * 100) if total else None,
                "f_share": (counts["F_count"] / total * 100) if total else None,
                "p_share": (counts["P_count"] / total * 100) if total else None,
                "coords_P_direct": float(sub["coords_opened_P"].sum()) if len(sub) else 0.0,
                "consolidations_attributed": (
                    float(sub["consolidations_attributed"].sum()) if len(sub) else 0.0
                ),
                "merged_coords_attributed": (
                    float(sub["merged_coords_attributed"].sum()) if len(sub) else 0.0
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    value_cols = [c for c in df.columns if c not in ("spec_id", "seed", "teacher")]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return _agg(df, ["spec_id", "teacher"], value_cols)


def damage_recovery(runs: list[RunData]) -> pd.DataFrame:
    """Forced-merge damage: eval macro-loss right after each forced merge vs
    right before, and steps until it recovers to the pre-merge level."""
    rows = []
    for run in runs:
        if run.events.empty or run.metrics.empty:
            continue
        merges = run.events[run.events["type"] == "consolidation"]
        if "forced" not in merges.columns:
            continue
        forced_steps = sorted(set(merges[merges["forced"] == True]["step"]))  # noqa: E712
        if not forced_steps:
            continue
        evals = run.metrics[run.metrics.get("eval/macro_loss").notna()][
            ["step", "eval/macro_loss"]
        ]
        for fstep in forced_steps:
            before = evals[evals["step"] < fstep]
            after = evals[evals["step"] >= fstep]
            if before.empty or after.empty:
                continue
            pre = float(before.iloc[-1]["eval/macro_loss"])
            post = float(after.iloc[0]["eval/macro_loss"])
            recovered = after[after["eval/macro_loss"] <= pre]
            rows.append(
                {
                    "spec_id": run.spec_id,
                    "seed": run.seed,
                    "forced_step": int(fstep),
                    "pre_merge_loss": pre,
                    "post_merge_loss": post,
                    "damage": post - pre,
                    "recovery_steps": (
                        int(recovered.iloc[0]["step"]) - int(fstep)
                        if len(recovered)
                        else None  # never recovered within the run
                    ),
                }
            )
    return pd.DataFrame(rows)

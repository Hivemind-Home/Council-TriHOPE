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
        if decisions.empty:
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
    if table.empty or baseline not in set(table["spec_id"]):
        return pd.DataFrame()
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


def p_selection_stats(runs: list[RunData]) -> pd.DataFrame:
    """When does P fire, and what do the signals look like at those moments?"""
    rows = []
    for run in runs:
        if run.events.empty:
            continue
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
    speed-of-learning comparison (P-on vs P-off after a consolidation)."""
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

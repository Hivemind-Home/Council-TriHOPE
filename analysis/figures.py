"""Matplotlib figures for the paper. Each function saves a PNG and returns
its path. All figures degrade gracefully when a run lacks the needed data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .loaders import RunData

_STORE_COLORS = {"R": "#4C78A8", "F": "#F58518", "P": "#54A24B"}


def _phase_shading(ax, run: RunData) -> None:
    bounds = run.phases
    for i, (start, name) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else None
        if i % 2 == 1:
            ax.axvspan(start, end or ax.get_xlim()[1], alpha=0.06, color="gray")
        ax.axvline(start, color="gray", lw=0.5, ls=":")


def plot_retention_curves(
    runs: list[RunData], out: Path, domains: Optional[list[str]] = None
) -> Optional[Path]:
    """Per-domain eval loss over steps, one line per spec."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False
    for run in runs:
        if run.metrics.empty:
            continue
        eval_cols = [
            c for c in run.metrics.columns
            if c.startswith("eval/") and c.endswith("/loss") and "bucket" not in c
        ]
        for col in eval_cols:
            domain = col.split("/")[1]
            if domains and domain not in domains:
                continue
            df = run.metrics[run.metrics[col].notna()]
            if df.empty:
                continue
            ax.plot(df["step"], df[col], label=f"{run.spec_id} {domain}", lw=1.2)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    _phase_shading(ax, runs[0])
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("Per-domain validation loss")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_action_composition(run: RunData, out: Path, window: int = 100) -> Optional[Path]:
    """Rolling R/F/P action shares over the stream (paper Figure 3 analogue)."""
    if run.events.empty:
        return None
    decisions = run.events[run.events["type"] == "decision"]
    if decisions.empty or "action" not in decisions.columns:
        return None  # controller-off runs emit no decisions
    shares = (
        pd.crosstab(decisions["step"], decisions["action"])
        .reindex(columns=["R", "F", "P"], fill_value=0)
        .rolling(window, min_periods=1)
        .mean()
    )
    shares = shares.div(shares.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.stackplot(
        shares.index,
        [shares[c] for c in ("R", "F", "P")],
        labels=["R", "F", "P"],
        colors=[_STORE_COLORS[c] for c in ("R", "F", "P")],
        alpha=0.85,
    )
    _phase_shading(ax, run)
    ax.set_xlabel("step")
    ax.set_ylabel("action share")
    ax.set_ylim(0, 1)
    ax.set_title(f"Action composition — {run.run_id}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_p_timeline(run: RunData, out: Path) -> Optional[Path]:
    """P decisions and consolidation events over the stream."""
    if run.events.empty or "action" not in run.events.columns:
        return None
    p_dec = run.events[
        (run.events["type"] == "decision") & (run.events["action"] == "P")
    ]
    merges = run.events[run.events["type"] == "consolidation"]
    if p_dec.empty and merges.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 2.8))
    if not p_dec.empty:
        ax.eventplot(p_dec["step"], lineoffsets=1.0, colors=_STORE_COLORS["P"], label="P decision")
    if not merges.empty:
        forced = merges[merges.get("forced", False) == True]  # noqa: E712
        organic = merges[merges.get("forced", False) != True]  # noqa: E712
        if len(organic):
            ax.eventplot(organic["step"], lineoffsets=0.0, colors="black", label="merge")
        if len(forced):
            ax.eventplot(forced["step"], lineoffsets=0.0, colors="red", label="forced merge")
    _phase_shading(ax, run)
    ax.set_yticks([0.0, 1.0], ["merge", "P decision"])
    ax.set_xlabel("step")
    ax.set_title(f"Permanent-memory timeline — {run.run_id}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def pareto_figure(df: pd.DataFrame, out: Path) -> Optional[Path]:
    """Figure 1: worst retention delta (left) and new-domain loss (right)
    against permanent writes, one marker per (spec, operating point),
    mean ± std over seeds. The x-axis is ``symlog`` because the LoRA-only
    controllers sit at exactly zero permanent writes.

    ``df`` is the output of :func:`analysis.tables.budget_curve`.
    """
    if df is None or df.empty or "permanent_writes_mean" not in df.columns:
        return None
    loss_cols = [c for c in df.columns if c.startswith("final_loss_") and c.endswith("_mean")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    specs = sorted(df["spec_id"].unique())
    cmap = plt.get_cmap("tab10")
    for i, spec in enumerate(specs):
        sub = df[df["spec_id"] == spec]
        x = sub["permanent_writes_mean"].to_numpy(dtype=float)
        xerr = sub.get("permanent_writes_std", pd.Series([0.0] * len(sub))).fillna(0).to_numpy()
        y = sub["worst_retention_delta_mean"].to_numpy(dtype=float)
        yerr = (
            sub.get("worst_retention_delta_std", pd.Series([0.0] * len(sub)))
            .fillna(0)
            .to_numpy()
        )
        axes[0].errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="o", color=cmap(i % 10), label=spec, capsize=2
        )
        for xi, yi, tag in zip(x, y, sub["threshold_tag"]):
            if tag:
                axes[0].annotate(str(tag), (xi, yi), fontsize=5, alpha=0.7)
        if loss_cols:
            new_loss = sub[loss_cols].mean(axis=1).to_numpy(dtype=float)
            axes[1].errorbar(x, new_loss, xerr=xerr, fmt="s", color=cmap(i % 10), label=spec)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1e3)
        ax.set_xlabel("permanent writes (coordinate·steps on base weights)")
    axes[0].set_ylabel("worst retention delta (↓)")
    axes[0].set_title("Forgetting vs. permanent-write budget")
    axes[1].set_ylabel("mean final eval loss (↓)")
    axes[1].set_title("Plasticity vs. permanent-write budget")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def containment_bars(df: pd.DataFrame, out: Path) -> Optional[Path]:
    """E5 containment: per method, where the corrupted teacher's actions
    went (R / F / P shares) and how many base coordinates it reached
    (directly + attributed merges). ``df`` is :func:`tables.containment`."""
    if df is None or df.empty or "r_share_mean" not in df.columns:
        return None
    specs = list(df["spec_id"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(specs))
    bottom = [0.0] * len(specs)
    for store in ("R", "F", "P"):
        vals = df[f"{store.lower()}_share_mean"].fillna(0).to_numpy(dtype=float)
        axes[0].bar(x, vals, bottom=bottom, color=_STORE_COLORS[store], label=store)
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[0].set_xticks(list(x), specs, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("share of the corrupted teacher's actions (%)")
    axes[0].set_title("Containment: where its updates went")
    axes[0].legend(fontsize=8)
    direct = df["coords_P_direct_mean"].fillna(0).to_numpy(dtype=float)
    merged = df["merged_coords_attributed_mean"].fillna(0).to_numpy(dtype=float)
    axes[1].bar(x, direct, color=_STORE_COLORS["P"], label="direct base writes")
    axes[1].bar(x, merged, bottom=direct, color="#B279A2", label="merged (attributed)")
    axes[1].set_xticks(list(x), specs, rotation=30, ha="right", fontsize=8)
    axes[1].set_yscale("symlog", linthresh=1e3)
    axes[1].set_ylabel("base coordinates reached")
    axes[1].set_title("Permanent footprint of the corrupted teacher")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_signal_traces(run: RunData, out: Path) -> Optional[Path]:
    """Mean routing signals over time from the metrics stream."""
    cols = {
        "signals/surprise_mean": "surprise S",
        "signals/repetition_mean": "repetition R",
        "signals/stability_C_mean": "cosine C",
        "signals/stability_adam_mean": "adam SNR",
    }
    if run.metrics.empty or not any(c in run.metrics.columns for c in cols):
        return None
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for col, label in cols.items():
        if col in run.metrics.columns:
            df = run.metrics[run.metrics[col].notna()]
            ax.plot(df["step"], df[col], label=label, lw=1.0)
    _phase_shading(ax, run)
    ax.set_xlabel("step")
    ax.set_ylabel("signal value")
    ax.set_title(f"Routing signals — {run.run_id}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

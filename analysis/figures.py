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
    if decisions.empty:
        return None
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
    if run.events.empty:
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

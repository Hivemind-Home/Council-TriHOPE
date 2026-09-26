#!/usr/bin/env python
"""Generate the paper's data figures (figures/*.pdf) from data/ via results_lib.

Palette: three validated categorical slots (dataviz reference palette, all-pairs
check passed on light surface) plus neutral grey for context marks. Families keep
the same colour in every figure:
  blue   = the method and its own variants
  aqua   = evidence-gated routing that never makes an update permanent
  orange = ungated / mis-gated permanence
  grey   = controls on other trainable surfaces
Every point is direct-labelled (aqua sits below 3:1 contrast -> labels are required);
hollow markers are single-seed points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_lib import DELTA_L, DELTA_M1, PHASES, ROOT, group_runs, load_runs, summarize, vals  # noqa: E402

FIG = ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

# (group key, label, family, label offset in points)
PARETO = [
    ("trihope", "controller (ref.)", "ours", (42, -6)),
    ("surprise_gate", "surprise-only", "noperm", (7, 0)),
    ("surprise_gate_s4", r"surprise-only $\theta_S{=}4$", "noperm", (7, -2)),
    ("no_consolidation", "never commit", "noperm", (7, 2)),
    ("frozen_blocks", "shared params only", "surface", (7, 2)),
    ("random_commit", "random commit", "ungated", (7, -2)),
    ("random_routing", "random relabel", "ungated", (7, 2)),
    ("p_only", "always commit", "ungated", (-6, 0)),
    ("no_cosine", r"no $\bar C$ gate", "ungated", (7, 0)),
    ("molf_style_a0p3", r"SNR $\tau{=}0.3$", "ungated", (-6, 0)),
    ("molf_style_a0p5", r"SNR $\tau{=}0.5$", "ungated", (7, 0)),
    ("molf_style_a0p7", r"SNR $\tau{=}0.7$", "ungated", (7, -3)),
    ("molf_style", "MoLF EPD", "ungated", (-6, 0)),
    ("full_ft", "full FT", "ungated", (0, 0)),
    ("lora_only", "LoRA only", "surface", (7, 0)),
    ("periodic_merge", "LoRA + periodic merge", "surface", (7, 0)),
]
OFFSET_B = {"trihope": (-6, -7), "random_routing": (7, 3), "molf_style_a0p5": (7, 3), "molf_style_a0p3": (-6, 0),
            "no_cosine": (24, -12)}
# Crowded points near the pool (many arms sit at M1 ~ 0.08) get fixed label positions in data
# coordinates, with a leader line back to the mark: key -> (x_text, y_text, ha).
TEXT_A = {
    "no_consolidation": (2.5e7, 0.121, "left"),
    "surprise_gate": (2.5e7, 0.107, "left"),
    "frozen_blocks": (2.5e7, 0.066, "left"),
    "surprise_gate_s4": (2.5e7, 0.052, "left"),
    "molf_style_a0p7": (4e10, 0.109, "left"),
    "random_commit": (4e10, 0.095, "left"),
    "trihope": (4e10, 0.068, "left"),
    "random_routing": (1.5e11, 0.182, "left"),
}
TEXT_B = {
    "trihope": (1.228, 0.066, "right"),
    "no_consolidation": (1.203, 0.124, "left"),
    "surprise_gate": (1.305, 0.112, "left"),
    "molf_style_a0p7": (1.305, 0.098, "left"),
    "frozen_blocks": (1.305, 0.080, "left"),
    "random_commit": (1.305, 0.068, "left"),
    "surprise_gate_s4": (1.305, 0.056, "left"),
}
# every point is labelled in both panels (panel-b offsets override where needed)
LABEL_B = {k for k, *_ in PARETO}
FAMILY = {
    "ours": (BLUE, "gated controller and its variants"),
    "noperm": (AQUA, "gated, never permanent"),
    "ungated": (ORANGE, "ungated / mis-gated permanence"),
    "surface": (GREY, "other trainable surfaces"),
}
Y_CAP = 0.33  # full fine-tuning (0.87, one seed diverged) is drawn as an off-scale arrow


def _mark(ax, x, y, xerr, yerr, color, n, label, off, clip=True, text_at=None):
    """One arm: error bars at the back, then leader line, then the marker, then the label.

    Layering keeps every point visible: error bars (zorder 1.5) and leader lines (3) are drawn
    beneath all markers (4), and labels (5) sit on a small white backing so a line or bar passing
    behind a label cannot run through its text.
    """
    hollow = n < 2
    if xerr or yerr:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=0, alpha=0.55, zorder=1.5)
    ax.plot(x, y, "o", ms=4.2 if not hollow else 3.8, mfc="white" if hollow else color,
            mec=color, mew=1.0, zorder=4)
    if not label:
        return
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
    if text_at is not None:
        xt, yt, ha = text_at
        ax.annotate("", (x, y), xytext=(xt, yt), textcoords="data", zorder=3,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.45, shrinkA=0, shrinkB=2.5))
        ax.text(xt, yt, label, fontsize=6.2, color=INK, va="center", ha=ha, zorder=5, bbox=box)
        return
    far = abs(off[0]) > 20  # long offset: draw a leader line back to the mark
    ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=6.2,
                color=INK, va="center", ha="left" if off[0] >= 0 else "right", zorder=5, bbox=box,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.45, shrinkA=1, shrinkB=2.5)
                if far else None)


def fig_pareto(groups: dict) -> None:
    pool = groups["trihope"]
    pm, psd = summarize(pool, "M1")["mean"], summarize(pool, "M1")["sd"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax in axes:
        ax.axhspan(pm - DELTA_M1, pm + DELTA_M1, color=BLUE, alpha=0.08, lw=0, zorder=0)
        ax.axhline(pm, color=BLUE, lw=0.6, alpha=0.5, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    # sweep cloud (single-seed threshold settings of the method), recessive
    sweep = [k for k in groups if k.startswith("trihope_s") and "_r0p" in k]
    for k in sweep:
        rs = groups[k]
        y = summarize(rs, "M1")["mean"]
        axes[0].plot(np.mean(vals(rs, "pw")), y, "+", ms=3.5, color=BLUE, mew=0.6,
                     alpha=0.7, zorder=2)
    for key, label, fam, off in PARETO:
        rs = groups.get(key)
        if not rs:
            continue
        color = FAMILY[fam][0]
        s1, sf = summarize(rs, "M1"), summarize(rs, "final")
        pw = float(np.mean(vals(rs, "pw")))
        y, ye = s1["mean"], s1.get("sd") or 0.0
        if y > Y_CAP:  # off-scale (full FT, one seed diverged): arrow at the top edge of (a) only
            med = summarize(rs, "M1")["median"]
            axes[0].annotate(f"{label}: mean {y:.2f}, median {med:.2f}", xy=(pw, Y_CAP), xytext=(pw / 40, Y_CAP - 0.012),
                             fontsize=6.2, color=INK, ha="right", va="center",
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                                             shrinkA=0, shrinkB=0))
            continue
        _mark(axes[0], pw, y, None, ye, color, len(rs), label, off, text_at=TEXT_A.get(key))
        _mark(axes[1], sf["mean"], y, sf.get("sd") or 0.0, ye, color, len(rs),
              label if key in LABEL_B else None, OFFSET_B.get(key, off), text_at=TEXT_B.get(key))
    ax = axes[0]
    ax.set_xscale("symlog", linthresh=1e7)
    ax.set_xlim(-2e6, 2e13)
    ax.set_xticks([0, 1e8, 1e10, 1e12])
    ax.set_xticklabels(["0", r"$10^{8}$", r"$10^{10}$", r"$10^{12}$"])
    ax.set_xlabel("routed permanent writes (coordinate$\\cdot$steps)")
    ax.set_ylabel(r"forgetting $M_1$ $\downarrow$")
    ax.set_ylim(0.04, Y_CAP)
    ax.set_title("(a) forgetting vs. permanence budget", loc="left")
    ax = axes[1]
    ax.set_xlabel(r"final validation loss (macro) $\downarrow$")
    ax.set_title("(b) forgetting vs. plasticity", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mfc=c, ms=4, label=t)
               for c, t in FAMILY.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=4,
                              label="single seed"))
    handles.append(plt.Line2D([], [], marker="+", ls="", color=BLUE, ms=4,
                              label="controller threshold sweep"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.2, columnspacing=1.2)
    fig.savefig(FIG / "pareto.pdf")
    plt.close(fig)


def _series(rs, key, steps=None):
    """Mean over runs of a metrics.csv column, aligned on logged steps."""
    table: dict[int, list[float]] = {}
    for r in rs:
        for row in r.metrics:
            v = row.get(key)
            if v in (None, ""):
                continue
            table.setdefault(int(float(row["step"])), []).append(float(v))
    xs = sorted(table)
    return np.array(xs), np.array([np.mean(table[x]) for x in xs])


def _phase_bands(ax, labels=True):
    short = {"general_warm": "warm", "code_recurrent": "code", "novel_inject": "novel",
             "math_recurrent": "math", "medical_recurrent": "medical",
             "code_revisit": "code\nrevisit", "mixed_tail": "mixed"}
    for i, (name, a, b) in enumerate(PHASES):
        if i % 2 == 0:
            ax.axvspan(a, b + 1, color="#f1f0ec", lw=0, zorder=0)
        if labels:
            ax.text((a + b) / 2, 1.02, short[name], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, color=INK2)


def fig_stream(groups: dict) -> None:
    pool_e1 = [r for r in groups["trihope"] if r.matrix.startswith("baselines")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.5, 1.95), sharex=True,
                                 gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.12})
    # (a) action shares: counts per logged step, averaged over seeds, 100-step rolling mean
    x, r = _series(pool_e1, "write/r_count")
    _, f = _series(pool_e1, "write/f_count")
    _, p = _series(pool_e1, "write/p_count")
    tot = np.maximum(r + f + p, 1e-9)
    k = 5
    ker = np.ones(k) / k
    sm = [np.convolve(v / tot, ker, mode="valid") for v in (f, r, p)]
    x = x[k // 2: len(x) - (k // 2)]
    _phase_bands(a1)
    a1.stackplot(x, sm[0], sm[1], sm[2], colors=["#d9d8d3", BLUE, ORANGE],
                 labels=[r"\textsc{withhold}", "defer", "commit"], lw=0, zorder=2)
    a1.set_ylim(0, 1)
    a1.set_ylabel("share of\nblock decisions")
    a1.set_yticks([0, 0.5, 1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#d9d8d3", BLUE, ORANGE)]
    a1.legend(handles, ["withhold", "defer", "commit"], loc="center left", ncol=1, frameon=False,
              bbox_to_anchor=(1.01, 0.5))
    for _n, a, _b in PHASES[1:]:
        a1.axvline(a, color="white", lw=0.8, zorder=3)
    # (b) code validation loss for three arms
    _phase_bands(a2, labels=False)
    for key, label, color in (("lora_only", "LoRA only", GREY), ("p_only", "always commit", ORANGE),
                              ("trihope", "controller (ref.)", BLUE)):
        rs = groups.get(key)
        if not rs:
            continue
        if key == "trihope":
            rs = pool_e1
        xs, ys = _series(rs, "eval/code/loss")
        a2.plot(xs, ys, color=color, lw=1.2 if key == "trihope" else 1.0, label=label, zorder=3)
    a2.set_ylabel("code val. loss")
    a2.set_xlabel("training step")
    a2.set_xlim(0, 6000)
    a2.legend(loc="center left", ncol=1, frameon=False, bbox_to_anchor=(1.01, 0.5))
    for ax in (a1, a2):
        ax.grid(False)
    fig.savefig(FIG / "stream.pdf")
    plt.close(fig)


ABL = [
    ("moments_optimizer", "optimizer's own (masked) moments"),
    ("topm_all", "select all blocks (top-M = all)"),
    ("no_cosine", r"no $\bar C$ (gate always passes)"),
    ("no_surprise", "no surprise"),
    ("stability_instant", r"instantaneous $C$ instead of $\bar C$"),
    ("no_teacher_conf", "no confidence gate / weighting"),
    ("topk_100", "top-K = 100% of ranks"),
    ("no_repetition", "no recurrence signal"),
    ("consolidation_strict", "strict consolidation"),
    ("no_volatility", "no volatility"),
    ("topk_25", "top-K = 25% of ranks"),
    ("topm_2", "top-M = 2"),
]


def fig_ablation(groups: dict) -> None:
    pool = groups["trihope"]
    pm1, pmf = summarize(pool, "M1")["mean"], summarize(pool, "final")["mean"]
    rows = [(k, lab) for k, lab in ABL if k in groups]
    rows.sort(key=lambda kv: summarize(groups[kv[0]], "M1")["mean"] - pm1)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95), sharey=True, gridspec_kw={"wspace": 0.06})
    ys = np.arange(len(rows))
    for ax, key, base, delta, title in (
        (axes[0], "M1", pm1, DELTA_M1, r"$\Delta$ forgetting $M_1$ vs. reference  ($\downarrow$ better)"),
        (axes[1], "final", pmf, DELTA_L, r"$\Delta$ final loss vs. reference  ($\downarrow$ better)"),
    ):
        ax.axvspan(-2 * delta, 2 * delta, color=BLUE, alpha=0.05, lw=0)
        ax.axvspan(-delta, delta, color=BLUE, alpha=0.08, lw=0)
        ax.axvline(0, color=BLUE, lw=0.6, alpha=0.6)
        for y, (k, _lab) in zip(ys, rows):
            s = summarize(groups[k], key)
            d = s["mean"] - base
            n = s["n"]
            ax.errorbar(d, y, xerr=s.get("sd") or 0.0, fmt="o", ms=3.8, color=BLUE,
                        mfc="white" if n < 2 else BLUE, mec=BLUE, mew=1.0, elinewidth=0.7)
        ax.set_title(title, loc="left", fontsize=7.2)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _k, lab in rows])
    axes[0].invert_yaxis()
    fig.savefig(FIG / "ablation.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- revision figures
EXPLORATORY = {"yoked_2x2_v1_live", "student17_v1_live", "order2_v1_live"}
SHORT = {
    "trihope": "controller (ref.)", "random_commit": "random commit", "random_routing": "random relabel",
    "no_cosine": r"no $\bar C$ gate", "p_only": "always commit", "molf_style": "MoLF EPD",
    "molf_style_a0p7": r"SNR $\tau{=}0.7$", "molf_style_a0p5": r"SNR $\tau{=}0.5$", "full_ft": "full FT",
    "no_retrieval": "no defer tier", "no_consolidation": "never commit", "surprise_gate": "surprise-only",
    "surprise_gate_s4": r"surprise-only $\theta_S{=}4$", "frozen_blocks": "shared only",
    "lora_only": "LoRA only", "periodic_merge": "periodic merge", "pgate_c0p35": r"$\theta_C{=}0.35$",
    "topm_all": "top-M = all", "moments_optimizer": "masked moments",
    "trihope_no_hash": "no bucket id",
}
FAM = {"trihope": "ours", "no_retrieval": "ours", "trihope_no_hash": "ours", "pgate_c0p35": "ours",
       "topm_all": "ours", "moments_optimizer": "ours",
       "no_consolidation": "noperm", "surprise_gate": "noperm", "surprise_gate_s4": "noperm",
       "random_commit": "ungated", "random_routing": "ungated", "no_cosine": "ungated", "p_only": "ungated",
       "molf_style": "ungated", "molf_style_a0p7": "ungated", "molf_style_a0p5": "ungated", "full_ft": "ungated",
       "frozen_blocks": "surface", "lora_only": "surface", "periodic_merge": "surface"}
WRITE_ARMS = ["trihope", "no_retrieval", "random_commit", "random_routing", "no_cosine", "pgate_c0p35",
              "molf_style_a0p7", "molf_style_a0p5", "p_only", "molf_style", "full_ft", "no_consolidation",
              "surprise_gate_s4", "frozen_blocks", "lora_only", "periodic_merge", "topm_all", "moments_optimizer"]


def _feats(groups):
    import revision
    from build_results import SHARED_PARAMS, SURFACE
    return {g: revision.arm_features(rs, SHARED_PARAMS, SURFACE.get(g)) for g, rs in groups.items()
            if not g.startswith("bt_") and g != "gold_ce"}


def _y(groups, g):
    s = summarize(groups[g], "M1")
    if s["mean"] > Y_CAP:            # full FT: plot the median, one seed diverged
        return s["median"], 0.0, True
    return s["mean"], s.get("sd") or 0.0, False


def _label(ax, x, y, text, dx=5, dy=0, ha="left"):
    if abs(dx) > 12 or abs(dy) > 10:  # far label: thin leader line, drawn beneath the markers
        ax.annotate("", (x, y), xytext=(dx, dy), textcoords="offset points", zorder=3,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=2))
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.8, color=INK,
                va="center", ha=ha, zorder=5,
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))


def _band(ax, groups):
    pm = summarize(groups["trihope"], "M1")["mean"]
    ax.axhspan(pm - DELTA_M1, pm + DELTA_M1, color=BLUE, alpha=0.08, lw=0, zorder=0)
    ax.axhline(pm, color=BLUE, lw=0.6, alpha=0.5, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig_writes(groups: dict) -> None:
    """Figure 2: forgetting against authorized, effective and realized routed writes."""
    F = _feats(groups)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharey=True, gridspec_kw={"wspace": 0.07})
    for ax in axes:
        _band(ax, groups)
    lab_a = {"trihope": (6, -7), "random_commit": (6, 6), "no_cosine": (6, 0), "p_only": (-5, 0),
             "molf_style": (-5, 0), "full_ft": (-5, 0), "periodic_merge": (6, -7), "topm_all": (-5, 0),
             "moments_optimizer": (6, 6), "lora_only": (6, 0), "random_routing": (-5, 0),
             "frozen_blocks": (6, -5), "no_consolidation": (6, 3)}
    lab_b = {"trihope": (6, -7), "no_cosine": (6, 0), "p_only": (-5, 0), "molf_style": (-5, 0),
             "full_ft": (-5, 0), "periodic_merge": (6, -7), "topm_all": (-5, 0), "moments_optimizer": (6, 6)}
    for g in WRITE_ARMS:
        if g not in groups:
            continue
        f, rs = F[g], groups[g]
        y, ye, med = _y(groups, g)
        col = FAMILY[FAM[g]][0]
        n = len(rs)
        for ax, x, labs in ((axes[0], f["total"], lab_a), (axes[1], f["base_effective"], lab_b)):
            ax.errorbar(x, y, yerr=ye or None, fmt="none", ecolor=col, elinewidth=0.7, alpha=0.55, zorder=1.5)
            ax.plot(x, y, "o", ms=3.8, mfc="white" if n < 3 else col, mec=col, mew=1.0, zorder=4)
            if g in labs:
                dx, dy = labs[g]
                _label(ax, x, y, SHORT[g] + (" (median)" if med else ""), dx, dy, "left" if dx > 0 else "right")
    # (c) realized routed drift, one point per run with a weight_drift.json
    import revision
    seen = set()
    for g in WRITE_ARMS:
        for r in groups.get(g, []):
            z = revision.realized(r)
            if not z:
                continue
            col = FAMILY[FAM[g]][0]
            axes[2].plot(z["routed_l2"], r.m["M1"], "o", ms=3.4, mfc=col, mec=col, alpha=0.85, zorder=4)
            if g not in seen:
                seen.add(g)
                off = {"trihope": (6, -7), "random_commit": (6, 7), "frozen_blocks": (6, 6),
                       "surprise_gate_s4": (6, -6), "molf_style_a0p7": (6, 8)}.get(g, (6, 0))
                _label(axes[2], z["routed_l2"], r.m["M1"], SHORT[g], *off)
    for ax, title, xl in ((axes[0], "(a) authorized", r"routed base writes (coord$\cdot$steps)"),
                          (axes[1], "(b) effective", r"commits + effective merges"),):
        ax.set_xscale("symlog", linthresh=1e7)
        ax.set_xlim(-2e6, 2e13)
        ax.set_xticks([0, 1e8, 1e10, 1e12])
        ax.set_xticklabels(["0", r"$10^{8}$", r"$10^{10}$", r"$10^{12}$"])
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
    axes[2].set_xscale("symlog", linthresh=1.0)
    axes[2].set_xlim(-0.2, 40)
    axes[2].set_xticks([0, 1, 3, 10, 30])
    axes[2].set_xticklabels(["0", "1", "3", "10", "30"])
    axes[2].set_title("(c) realized (runs with a weight diff)", loc="left")
    axes[2].set_xlabel(r"routed base drift $\|\theta_T-\theta_0\|_2$")
    axes[0].set_ylabel(r"forgetting $M_1$ $\downarrow$")
    axes[0].set_ylim(0.04, Y_CAP)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mfc=c, ms=4, label=t) for c, t in FAMILY.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=4, label="fewer than 3 runs"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.2, columnspacing=0.9)
    fig.savefig(FIG / "writes.pdf")
    plt.close(fig)


def fig_volume(groups: dict) -> None:
    """Figure 3: forgetting vs final loss; vs all non-shared write volume; vs replayed rows."""
    F = _feats(groups)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharey=True, gridspec_kw={"wspace": 0.07})
    for ax in axes:
        _band(ax, groups)
    keys = [g for g in F if F[g]["M1"] is not None and not g.startswith("plateau")]
    labs = [
        {"trihope": (-5, -7), "lora_only": (6, 0), "topm_all": (6, -24),
         "p_only": (6, 0), "molf_style": (6, 0), "no_cosine": (6, -3),
         "random_routing": (6, 3), "frozen_blocks": (6, -2), "surprise_gate_s4": (6, -7),
         "full_ft": (-5, 0), "no_retrieval": (6, 3)},
        {"trihope": (18, -7), "lora_only": (-5, -3), "topm_all": (-5, 3), "moments_optimizer": (6, 6),
         "periodic_merge": (6, 0), "p_only": (-5, 0), "molf_style": (-5, 0), "full_ft": (-5, 0),
         "frozen_blocks": (6, 5), "surprise_gate_s4": (6, -7), "no_cosine": (6, 0)},
        {"trihope": (6, -8), "moments_optimizer": (-5, 5), "topm_all": (-5, 0), "random_routing": (6, 0),
         "surprise_gate_s4": (6, -6)},
    ]
    for g in keys:
        rs = groups[g]
        n = len(rs)
        y, ye, med = _y(groups, g)
        col = FAMILY[FAM[g]][0] if g in FAM else GREY
        big = g in FAM
        vol = F[g]["vol"]
        fl = summarize(rs, "final")["mean"]
        rep = F[g]["replay"]
        for ax, x, lab in ((axes[0], fl, labs[0]), (axes[1], vol, labs[1]), (axes[2], rep, labs[2])):
            if ax is axes[0] and not big:
                continue
            ax.plot(x, y, "o", ms=3.8 if big else 2.4, mfc=(col if n >= 3 else "white") if big else "#c9c8c3",
                    mec=col if big else "#c9c8c3", mew=1.0 if big else 0.5, zorder=4 if big else 2)
            if g in lab and big:
                dx, dy = lab[g]
                txt = "periodic\nmerge" if (ax is axes[1] and g == "periodic_merge") else SHORT[g]
                _label(ax, x, y, txt + (" (median)" if med else ""), dx, dy, "left" if dx > 0 else "right")
    axes[0].set_title("(a) forgetting vs. plasticity", loc="left")
    axes[0].set_xlabel(r"final validation loss $\downarrow$")
    axes[1].set_xscale("symlog", linthresh=1e7)
    axes[1].set_xlim(-2e6, 2e13)
    axes[1].set_xticks([0, 1e8, 1e10, 1e12])
    axes[1].set_xticklabels(["0", r"$10^{8}$", r"$10^{10}$", r"$10^{12}$"])
    axes[1].set_title("(b) all non-shared writes", loc="left")
    axes[1].set_xlabel(r"effective base + adapter coord$\cdot$steps")
    axes[2].set_xscale("symlog", linthresh=10)
    axes[2].set_xlim(-2, 5e3)
    axes[2].set_xticks([0, 10, 100, 1000])
    axes[2].set_xticklabels(["0", "10", "100", "1000"])
    axes[2].set_title("(c) replay volume", loc="left")
    axes[2].set_xlabel("replayed rows")
    axes[0].set_ylabel(r"forgetting $M_1$ $\downarrow$")
    axes[0].set_ylim(0.04, Y_CAP)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mfc=c, ms=4, label=t) for c, t in FAMILY.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=4, label="fewer than 3 runs"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#c9c8c3", mfc="#c9c8c3", ms=3, label="threshold sweeps"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.2, columnspacing=0.8)
    fig.savefig(FIG / "volume.pdf")
    plt.close(fig)


def fig_stats(groups: dict) -> None:
    """Appendix figure: (a) M1 of the reference pool by stream seed; (b) theta_C dose-response;
    (c) leave-one-out residuals of the write-volume regression (post hoc)."""
    import revision
    F = _feats(groups)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95), gridspec_kw={"wspace": 0.45})
    pool = groups["trihope"]
    seeds = (1337, 2024, 7)
    for i, s in enumerate(seeds):
        ys = [r.m["M1"] for r in pool if r.seed == s]
        axes[0].plot([i] * len(ys), ys, "o", ms=3.5, mfc=BLUE, mec=BLUE, alpha=0.8)
        axes[0].plot([i - 0.2, i + 0.2], [np.mean(ys)] * 2, color=INK, lw=0.9)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels([str(s) for s in seeds])
    axes[0].set_xlim(-0.6, 2.6)
    axes[0].set_xlabel("stream seed")
    axes[0].set_ylabel(r"$M_1$ (reference pool)")
    axes[0].set_title("(a) reruns by seed", loc="left")
    # (b) dose-response of the agreement gate
    pts = [("no_consolidation", "never"), ("pgate_c0p35", "0.35"), ("trihope", "0.5"), ("no_cosine", "off")]
    xs = []
    for i, (g, lab) in enumerate(pts):
        if g not in groups:
            continue
        s = summarize(groups[g], "M1")
        n = s["n"]
        axes[1].errorbar(i, s["mean"], yerr=s.get("sd") or None, fmt="o", ms=3.8, color=BLUE,
                         mfc="white" if n < 3 else BLUE, mec=BLUE, elinewidth=0.7)
        xs.append(lab)
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels(xs)
    axes[1].set_xlabel(r"commit threshold $\theta_C$")
    axes[1].set_ylabel(r"$M_1$")
    axes[1].set_title(r"(b) $\bar C$ gate", loc="left")
    # (c) LOO residuals of the all-configuration regression, sorted
    import math as _m
    gs = [g for g in F if F[g]["M1"] is not None and not g.startswith("plateau") and g != "full_ft"]
    X = np.array([[_m.log10(F[g][c] + 1) for c in ("commit", "eff_merge_cs", "adapter_cs", "replay")] for g in gs])
    y = np.array([F[g]["M1"] for g in gs])
    reg = revision.ols(X, y, ["commit", "eff_merge_cs", "adapter_cs", "replay"])
    order = np.argsort(reg["loo"])
    cols = [ORANGE if gs[j] in ("topm_all", "moments_optimizer") else (BLUE if gs[j] == "trihope" else GREY)
            for j in order]
    axes[2].bar(range(len(order)), reg["loo"][order], color=cols, width=0.8)
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
                   labels=["top-M = all, masked moments", "controller (ref.)"], frameon=False,
                   fontsize=5.4, loc="upper left", handlelength=0.8)
    axes[2].axhline(0, color=INK2, lw=0.5)
    axes[2].set_xticks([])
    axes[2].set_xlabel("configurations (sorted)")
    axes[2].set_ylabel(r"LOO residual in $M_1$")
    axes[2].set_title("(c) regression residuals", loc="left")
    fig.savefig(FIG / "stats.pdf")
    plt.close(fig)



def main() -> None:
    FIG.mkdir(exist_ok=True)
    runs = [r for r in load_runs("live") if r.matrix not in EXPLORATORY]
    groups = group_runs(runs)
    # fig_pareto (old Figure 2) is superseded by fig_writes + fig_volume
    # stream.pdf is now a hand-drawn figure; its values were checked against this series
    # and numbers.tex (commitEarlyPct, earlyWindowPct, recov). fig_stream(groups) would overwrite it.
    fig_ablation(groups)
    fig_writes(groups)
    fig_volume(groups)
    fig_stats(groups)
    print("figures:", ", ".join(p.name for p in sorted(FIG.glob("*.pdf"))))


if __name__ == "__main__":
    main()

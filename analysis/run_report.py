"""Generate the full analysis bundle for one experiment.

    python -m analysis.run_report runs/baselines_small_v1 [--out DIR]

Writes CSV tables, PNG figures, and a report.md summary into
``{exp_dir}/analysis`` (or ``--out``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import figures, tables
from .loaders import RunData, load_experiment


def generate_report(exp_dir: Path, out_dir: Path) -> Path:
    runs = load_experiment(exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [f"# Report — {exp_dir.name}", ""]
    sections.append(
        f"{len(runs)} runs: " + ", ".join(sorted({r.run_id for r in runs}))
    )
    sections.append("")

    table_builders = {
        "forgetting_table": tables.forgetting_table,
        "action_share_by_phase": tables.action_share_by_phase,
        "ablation_deltas": tables.ablation_deltas,
        "p_selection_stats": tables.p_selection_stats,
        "adapter_reuse_aulc": tables.adapter_reuse_aulc,
        "damage_recovery": tables.damage_recovery,
    }
    for name, builder in table_builders.items():
        try:
            df = builder(runs)
        except Exception as exc:  # noqa: BLE001 — a partial report beats none
            sections.append(f"## {name}\n\n_error: {exc}_\n")
            continue
        if df is None or df.empty:
            sections.append(f"## {name}\n\n_no data_\n")
            continue
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        sections.append(f"## {name}\n")
        sections.append(df.round(4).to_markdown(index=False))
        sections.append("")

    fig_paths: list[Path] = []
    ret = figures.plot_retention_curves(runs, out_dir / "retention_curves.png")
    if ret:
        fig_paths.append(ret)
    for run in runs:
        for fn, suffix in (
            (figures.plot_action_composition, "actions"),
            (figures.plot_p_timeline, "p_timeline"),
            (figures.plot_signal_traces, "signals"),
        ):
            path = fn(run, out_dir / f"{run.run_id}_{suffix}.png")
            if path:
                fig_paths.append(path)

    if fig_paths:
        sections.append("## Figures\n")
        for path in fig_paths:
            sections.append(f"![{path.stem}]({path.name})")
        sections.append("")

    report = out_dir / "report.md"
    report.write_text("\n".join(sections))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out or (args.exp_dir / "analysis")
    report = generate_report(args.exp_dir, out_dir)
    print(f"Report written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the full analysis bundle for one or more experiments.

    python -m analysis.run_report runs/baselines_small_v1 [runs/other_v1 ...] [--out DIR]

Writes CSV tables, PNG figures, and a report.md summary into
``{exp_dir}/analysis`` (or ``--out``). With several experiment directories
the runs are pooled and every spec id is prefixed with its experiment name
(``baselines_small_v1/trihope``) so E1 and the threshold sweep can be
plotted on one budget curve without run-id collisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import figures, tables
from .loaders import RunData, load_experiment


def _load_all(exp_dirs: list[Path]) -> list[RunData]:
    runs: list[RunData] = []
    for exp_dir in exp_dirs:
        loaded = load_experiment(exp_dir)
        if len(exp_dirs) > 1:
            for run in loaded:
                run.spec_id = f"{exp_dir.name}/{run.spec_id}"
                run.run_id = f"{exp_dir.name}/{run.run_id}"
        runs.extend(loaded)
    return runs


def generate_report(exp_dir: "Path | list[Path]", out_dir: Path) -> Path:
    exp_dirs = [Path(d) for d in (exp_dir if isinstance(exp_dir, (list, tuple)) else [exp_dir])]
    runs = _load_all(exp_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)

    title = ", ".join(d.name for d in exp_dirs)
    sections: list[str] = [f"# Report — {title}", ""]
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
        "budget_curve": tables.budget_curve,
    }
    built: dict[str, object] = {}
    for name, builder in table_builders.items():
        try:
            df = builder(runs)
        except Exception as exc:  # noqa: BLE001 — a partial report beats none
            sections.append(f"## {name}\n\n_error: {exc}_\n")
            continue
        if df is None or df.empty:
            sections.append(f"## {name}\n\n_no data_\n")
            continue
        built[name] = df
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        sections.append(f"## {name}\n")
        sections.append(df.round(4).to_markdown(index=False))
        sections.append("")

    fig_paths: list[Path] = []
    ret = figures.plot_retention_curves(runs, out_dir / "retention_curves.png")
    if ret:
        fig_paths.append(ret)
    if "budget_curve" in built:
        try:
            pareto = figures.pareto_figure(built["budget_curve"], out_dir / "pareto_budget.png")
        except Exception as exc:  # noqa: BLE001
            sections.append(f"_pareto_budget.png failed: {exc}_\n")
            pareto = None
        if pareto:
            fig_paths.append(pareto)
    for run in runs:
        safe_id = run.run_id.replace("/", "__")
        for fn, suffix in (
            (figures.plot_action_composition, "actions"),
            (figures.plot_p_timeline, "p_timeline"),
            (figures.plot_signal_traces, "signals"),
        ):
            path = fn(run, out_dir / f"{safe_id}_{suffix}.png")
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
    parser.add_argument("exp_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out or (args.exp_dirs[0] / "analysis")
    report = generate_report(args.exp_dirs, out_dir)
    print(f"Report written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

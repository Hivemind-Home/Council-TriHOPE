"""CLI entry points for hivemind."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(help="Hivemind: Multi-Teacher Distillation with R/F/P Routing")


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Check runtime capabilities."""
    import torch

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if json_output:
        import json

        typer.echo(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            typer.echo(f"  {k}: {v}")


@app.command()
def smoke(config_name: str = typer.Option("pilot_smoke", help="Config name")) -> None:
    """Run a quick smoke test."""

    # Use hydra compose API for programmatic access
    from hydra import compose, initialize_config_dir

    from .config_utils import unwrap_config
    from .device import resolve_device
    from .training import run_training_loop

    config_dir = str(Path(__file__).parent.parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        cfg = unwrap_config(cfg)
        device = resolve_device(cfg.train.device)
        run_training_loop(cfg, device=device)


@app.command()
def experiment(
    manifest: Path = typer.Argument(..., help="Experiment manifest YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the run matrix and exit"),
    resume: bool = typer.Option(False, "--resume", help="Skip done runs, resume failed ones"),
    only: str = typer.Option(None, "--only", help="Run only this spec id"),
    max_hours: float = typer.Option(None, "--max-hours", help="Stop launching past this budget"),
) -> None:
    """Run an experiment matrix (see scripts/run_experiment.py)."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from run_experiment import main as run_main

    argv = [str(manifest)]
    if dry_run:
        argv.append("--dry-run")
    if resume:
        argv.append("--resume")
    if only:
        argv.extend(["--only", only])
    if max_hours is not None:
        argv.extend(["--max-hours", str(max_hours)])
    raise SystemExit(run_main(argv))


@app.command()
def train(config_name: str = typer.Option("pilot", help="Config name")) -> None:
    """Run training."""
    from hydra import compose, initialize_config_dir

    from .config_utils import unwrap_config
    from .device import resolve_device
    from .training import run_training_loop

    config_dir = str(Path(__file__).parent.parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        cfg = unwrap_config(cfg)
        device = resolve_device(cfg.train.device)
        run_training_loop(cfg, device=device)


if __name__ == "__main__":
    app()

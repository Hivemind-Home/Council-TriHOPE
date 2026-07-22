#!/usr/bin/env python
"""Experiment matrix runner: manifest → sequential subprocess runs.

A manifest (``configs/experiments/*.yaml``) declares a base config, a list of
run specs (id + Hydra overrides), and seeds. Each (run, seed) executes as its
own ``python train.py`` subprocess (OOM/CUDA-fragmentation isolation) with a
private run directory:

    runs/{experiment}/{run_id}-seed{seed}/
        hydra/            # Hydra config snapshot for the run
        metrics.jsonl     # per-step metrics
        events.jsonl      # routing/consolidation event trace
        run_summary.json  # memory/time/retention totals
        checkpoints/      # pruned per cleanup policy
        stdout.log        # tee'd subprocess output
        status.json       # pending | running | done | failed

Usage:
    python scripts/run_experiment.py configs/experiments/baselines_small.yaml
    python scripts/run_experiment.py MANIFEST --dry-run          # print the matrix
    python scripts/run_experiment.py MANIFEST --resume           # skip done, resume failed
    python scripts/run_experiment.py MANIFEST --only trihope     # one spec id
    python scripts/run_experiment.py MANIFEST --max-hours 8      # stop launching after budget

Manifest schema:
    experiment: baselines_small_v1
    base_config: stream_small
    output_root: runs
    seeds: [1337]
    cleanup_checkpoints: keep_final      # all | keep_final | keep_none
    runs:
      - {id: trihope, overrides: []}
      - {id: full_ft, overrides: [controller.enabled=false, model.lora.rank=0]}
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ResolvedRun:
    run_id: str          # "{spec_id}-seed{seed}" — parsed by analysis for aggregation
    spec_id: str
    seed: int
    run_dir: Path
    overrides: list[str] = field(default_factory=list)


def load_manifest(path: Path) -> dict:
    manifest = yaml.safe_load(path.read_text())
    for key in ("experiment", "base_config", "runs"):
        if key not in manifest:
            raise ValueError(f"Manifest missing required key '{key}'")
    return manifest


def expand_matrix(manifest: dict) -> list[ResolvedRun]:
    """Expand (runs × seeds) into concrete run dirs + override lists."""
    out_root = Path(manifest.get("output_root", "runs")) / manifest["experiment"]
    seeds = [int(s) for s in manifest.get("seeds", [1337])]
    resolved: list[ResolvedRun] = []
    seen_ids: set[str] = set()
    for spec in manifest["runs"]:
        spec_id = str(spec["id"])
        if spec_id in seen_ids:
            raise ValueError(f"Duplicate run id '{spec_id}' in manifest")
        seen_ids.add(spec_id)
        spec_overrides = [str(o) for o in spec.get("overrides", [])]
        for seed in seeds:
            run_id = f"{spec_id}-seed{seed}"
            run_dir = out_root / run_id
            # ``++`` = add-or-override, so forced keys work whether or not the
            # base config declares them.
            forced = [
                f"++train.seed={seed}",
                f"++checkpoint.dir={run_dir / 'checkpoints'}",
                f"++logging.path={run_dir / 'metrics.jsonl'}",
                f"++logging.events_path={run_dir / 'events.jsonl'}",
                f"++run.dir={run_dir}",
                f"++hydra.run.dir={run_dir / 'hydra'}",
            ]
            resolved.append(
                ResolvedRun(
                    run_id=run_id,
                    spec_id=spec_id,
                    seed=seed,
                    run_dir=run_dir,
                    overrides=spec_overrides + forced,
                )
            )
    return resolved


def _read_status(run: ResolvedRun) -> dict:
    path = run.run_dir / "status.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"state": "pending"}


def _write_status(run: ResolvedRun, state: str, **extra) -> None:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "run_id": run.run_id, "updated": time.time(), **extra}
    (run.run_dir / "status.json").write_text(json.dumps(payload, indent=2))


def launch(run: ResolvedRun, base_config: str, resume: bool = False) -> int:
    """Run one training subprocess, tee-ing output to stdout.log."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--config-name",
        base_config,
        *run.overrides,
    ]
    if resume and (run.run_dir / "checkpoints").exists():
        cmd.append("++checkpoint.resume_from=latest")

    run.run_dir.mkdir(parents=True, exist_ok=True)
    _write_status(run, "running", cmd=" ".join(cmd), started=time.time())
    print(f"[{run.run_id}] {' '.join(cmd)}")

    with open(run.run_dir / "stdout.log", "a", encoding="utf-8") as log:
        log.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT
        )
        code = proc.wait()

    # A run only counts as done when its summary was written.
    ok = code == 0 and (run.run_dir / "run_summary.json").exists()
    _write_status(run, "done" if ok else "failed", exit_code=code)
    return code


def cleanup_checkpoints(run: ResolvedRun, policy: str) -> None:
    ckpt_dir = run.run_dir / "checkpoints"
    if policy == "all" or not ckpt_dir.exists():
        return
    step_dirs = sorted(
        p for p in ckpt_dir.iterdir() if p.is_dir() and not p.is_symlink()
    )
    if policy == "keep_none":
        doomed = step_dirs
    elif policy == "keep_final":
        # Keep the newest untagged dir + all tagged (pre_merge rollback) dirs.
        untagged = [p for p in step_dirs if "_" not in p.name.removeprefix("step_")]
        doomed = untagged[:-1]
    else:
        raise ValueError(f"Unknown cleanup policy '{policy}'")
    for p in doomed:
        shutil.rmtree(p, ignore_errors=True)


def write_index(manifest: dict, runs: list[ResolvedRun]) -> None:
    out_root = Path(manifest.get("output_root", "runs")) / manifest["experiment"]
    out_root.mkdir(parents=True, exist_ok=True)
    index = {
        "experiment": manifest["experiment"],
        "base_config": manifest["base_config"],
        "updated": time.time(),
        "runs": [
            {
                "run_id": r.run_id,
                "spec_id": r.spec_id,
                "seed": r.seed,
                "dir": str(r.run_dir),
                **_read_status(r),
            }
            for r in runs
        ],
    }
    (out_root / "manifest_index.json").write_text(json.dumps(index, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    parser.add_argument("--resume", action="store_true", help="skip done runs, resume failed ones")
    parser.add_argument("--only", type=str, default=None, help="run only this spec id")
    parser.add_argument(
        "--max-hours", type=float, default=None, help="stop launching past this budget"
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    runs = expand_matrix(manifest)
    if args.only:
        runs = [r for r in runs if r.spec_id == args.only]
        if not runs:
            print(f"No runs match --only {args.only}")
            return 2

    if args.dry_run:
        for r in runs:
            print(f"{r.run_id:40s} {_read_status(r)['state']:8s} {' '.join(r.overrides)}")
        return 0

    policy = str(manifest.get("cleanup_checkpoints", "keep_final"))
    budget_start = time.time()
    failures = 0
    for run in runs:
        status = _read_status(run)
        if args.resume and status["state"] == "done":
            print(f"[{run.run_id}] already done — skipping")
            continue
        if args.max_hours is not None:
            elapsed_h = (time.time() - budget_start) / 3600
            if elapsed_h > args.max_hours:
                print(f"Budget of {args.max_hours}h exhausted — stopping before {run.run_id}")
                break
        resume_this = args.resume and status["state"] in ("running", "failed")
        code = launch(run, manifest["base_config"], resume=resume_this)
        if code == 0:
            cleanup_checkpoints(run, policy)
        else:
            failures += 1
            print(f"[{run.run_id}] FAILED (exit {code}) — continuing with next run")
        write_index(manifest, runs)

    write_index(manifest, runs)
    print(f"\nDone: {sum(1 for r in runs if _read_status(r)['state'] == 'done')}/{len(runs)} "
          f"runs complete, {failures} failed this session.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

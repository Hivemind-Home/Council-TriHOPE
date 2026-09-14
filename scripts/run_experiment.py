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
import os
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


def _force_override(override: str) -> str:
    """Make a manifest override add-or-override (``++key=value``).

    Hydra's struct mode rejects ``key=value`` for a key the base config does
    not declare, and most baseline knobs (``controller.policy.mode``,
    ``controller.consolidation.trigger``, the plateau window, ...) are
    dataclass defaults that the stream YAMLs never spell out. ``++`` sets the
    key whether or not it exists; a misspelt key still fails fast because
    the controller dataclasses reject unknown fields at construction.
    ``+``/``++``/``~`` prefixes written by hand are left alone.
    """
    return override if override[:1] in "+~" else f"++{override}"


#: What ``--live`` adds to every spec: four frozen Qwen-vocabulary teachers
#: (``teachers.pretrained`` in the stream YAML) score the cached text and the
#: student matches their distributions; strict routing so an unregistered
#: domain fails instead of distilling through teacher 0.
LIVE_OVERRIDES = (
    "teachers.mode=live",
    "teachers.num_teachers=4",
    "distillation.lambda_ce=0.5",
    "data.router_strict=true",
)


def apply_live(manifest: dict) -> dict:
    """Return a copy of ``manifest`` rewritten for live teachers.

    The experiment name gains a ``_live`` suffix (its own ``runs/`` directory,
    never pooled with cache-mode results), every spec gets
    :data:`LIVE_OVERRIDES`, and any override that points into the cache-mode
    experiment directory (random_routing's action-share file) is redirected
    to the live one.
    """
    import copy

    live = copy.deepcopy(manifest)
    old_name = str(live["experiment"])
    new_name = old_name if old_name.endswith("_live") else f"{old_name}_live"
    live["experiment"] = new_name
    for spec in live.get("runs", []):
        overrides = list(spec.get("overrides", []))
        overrides = [
            (
                [str(x).replace(f"runs/{old_name}/", f"runs/{new_name}/") for x in o]
                if isinstance(o, (list, tuple))
                else str(o).replace(f"runs/{old_name}/", f"runs/{new_name}/")
            )
            for o in overrides
        ]
        spec["overrides"] = list(LIVE_OVERRIDES) + overrides
    return live


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
        # A YAML anchor (``- *common``) splices a nested list; flatten one level.
        spec_overrides: list[str] = []
        for o in spec.get("overrides", []):
            if isinstance(o, (list, tuple)):
                spec_overrides.extend(_force_override(str(x)) for x in o)
            else:
                spec_overrides.append(_force_override(str(o)))
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
                # Per-rank: torchrun sets RANK, and N processes writing the
                # same .hydra/ snapshot concurrently is a race. Analysis only
                # ever reads run.dir, so per-rank snapshots are harmless.
                f"++hydra.run.dir={run_dir / 'hydra'}/rank${{oc.env:RANK,0}}",
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


def build_command(run: ResolvedRun, base_config: str, nproc: int, resume: bool) -> list[str]:
    """Plain python for one process, torch.distributed.run for many.

    ``python -m torch.distributed.run`` rather than the ``torchrun`` shim so
    the same interpreter is guaranteed (this repo runs under conda/uv where
    PATH resolution is not reliable). ``distributed.world_size`` is forwarded
    so init_distributed can cross-check it against WORLD_SIZE and fail loudly
    on a mismatch instead of silently training on fewer GPUs.
    """
    if nproc > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", f"--nproc_per_node={nproc}",
            # Per-rank stdout/stderr files; without this N ranks interleave
            # into one fd and the log is unreadable.
            f"--log-dir={run.run_dir / 'torchrun_logs'}",
            "--redirects=3", "--tee=3",
            str(REPO_ROOT / "train.py"),
            "--config-name", base_config,
            *run.overrides,
            "++distributed.enabled=true",
            f"++distributed.world_size={nproc}",
        ]
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "train.py"),
            "--config-name",
            base_config,
            *run.overrides,
        ]
    if resume and (run.run_dir / "checkpoints").exists():
        cmd.append("++checkpoint.resume_from=latest")
    return cmd


def launch(
    run: ResolvedRun,
    base_config: str,
    resume: bool = False,
    nproc: int = 1,
    gpu: int | None = None,
) -> int:
    """Run one training subprocess, tee-ing output to stdout.log."""
    cmd = build_command(run, base_config, nproc, resume)

    env = dict(os.environ)
    if gpu is not None:
        gpu = _visible_device(gpu)
        # One spec per GPU (--parallel-gpus). For a matrix of INDEPENDENT
        # runs this beats DDP-ing a single spec across N GPUs: no
        # collectives, no batch-size divisibility constraint, no stream
        # digest change, and N times the runs rather than N times the
        # tokens on one run.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    run.run_dir.mkdir(parents=True, exist_ok=True)
    _write_status(run, "running", cmd=" ".join(cmd), started=time.time())
    print(f"[{run.run_id}]{f' gpu={gpu}' if gpu is not None else ''} {' '.join(cmd)}")

    with open(run.run_dir / "stdout.log", "a", encoding="utf-8") as log:
        log.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env
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


def _device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001 — torch may be absent in a dry-run env
        return 0


def _visible_device(slot: int) -> str:
    """Map a 0-based slot to a device id the parent actually owns.

    A bare index is wrong whenever the parent already has a restricted
    allocation (Slurm, Lightning): with CUDA_VISIBLE_DEVICES=4,5 the
    children would be pointed at physical GPUs 0 and 1, which this job does
    not own.
    """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not inherited:
        return str(slot)
    ids = [x.strip() for x in inherited.split(",") if x.strip()]
    return ids[slot] if slot < len(ids) else str(slot)


def _check_shardable(manifest: dict, nproc: int) -> int:
    """data.batch_size is the GLOBAL batch and must divide by the rank count.

    Checked here so the failure names the fix once, instead of every rank
    crashing inside StreamBatchSampler. stream_headline ships batch_size: 1
    precisely because it is tuned for one GPU.
    """
    import yaml as _yaml

    cfg_path = REPO_ROOT / "configs" / f"{manifest['base_config']}.yaml"
    try:
        batch_size = int(_yaml.safe_load(cfg_path.read_text())["data"]["batch_size"])
    except Exception:  # noqa: BLE001 — an unreadable config is preflight's problem
        return 0
    if batch_size % nproc == 0:
        return 0
    print(
        f"data.batch_size={batch_size} in {manifest['base_config']} is the GLOBAL "
        f"batch and does not divide by --nproc-per-node {nproc}.\n"
        f"  Add an override to keep the per-device batch at {batch_size}:\n"
        f"      overrides: [data.batch_size={batch_size * nproc}]\n"
        f"  NOTE: changing batch_size changes the stream digest, so this is a "
        f"FRESH run — existing checkpoints will not resume."
    )
    return 2


def _preflight_manifest(manifest: dict) -> int:
    """Metadata-only validation of the manifest's base config."""
    cmd = [
        sys.executable, "-m", "hivemind", "preflight",
        "--config-name", str(manifest["base_config"]), "--metadata-only",
    ]
    print(f"[preflight] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode:
        print("[preflight] FAILED — refusing to launch. Use --no-preflight to override.")
    return proc.returncode


def _run_parallel(
    runs, manifest, args, policy: str, nproc: int, share_gpu: bool = False
) -> int:
    """N specs concurrently: one per GPU (``--parallel-gpus``), or N processes
    sharing the single visible GPU (``--concurrent``, ``share_gpu=True`` —
    the 0.6B student uses a small fraction of a large card). A slot is
    refilled as soon as its run finishes."""
    from concurrent.futures import ThreadPoolExecutor

    if share_gpu:
        n = int(args.concurrent)
    else:
        n = int(args.parallel_gpus)
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        available = (
            len([x for x in inherited.split(",") if x.strip()])
            if inherited
            else _device_count()
        )
        if available and n > available:
            print(
                f"--parallel-gpus {n} exceeds the {available} GPU(s) this process can "
                f"see; capping to {available}."
            )
            n = available
    pending = []
    for run in runs:
        status = _read_status(run)
        if args.resume and status["state"] == "done":
            print(f"[{run.run_id}] already done — skipping")
            continue
        pending.append((run, args.resume and status["state"] in ("running", "failed")))

    print(
        f"Dispatching {len(pending)} runs "
        + (f"{n} at a time on the shared GPU..." if share_gpu else f"across {n} GPUs...")
    )
    failures = 0
    # A free-GPU queue rather than a static split, so a short run does not
    # leave its GPU idle while a long one on another slot finishes.
    free_gpus: list[int] = list(range(n))
    lock = __import__("threading").Lock()

    budget_start = time.time()

    def _work(item):
        run, resume_this = item
        # --max-hours was only honoured by the sequential loop, so a
        # --parallel-gpus session ignored the budget entirely.
        if args.max_hours is not None:
            if (time.time() - budget_start) / 3600 > args.max_hours:
                print(f"Budget of {args.max_hours}h exhausted — skipping {run.run_id}")
                return run, 0
        with lock:
            gpu = free_gpus.pop()
        try:
            return run, launch(
                run,
                manifest["base_config"],
                resume=resume_this,
                nproc=nproc,
                # Shared-GPU mode inherits the parent's device visibility.
                gpu=None if share_gpu else gpu,
            )
        finally:
            with lock:
                free_gpus.append(gpu)

    try:
        with ThreadPoolExecutor(max_workers=n) as pool:
            for run, code in pool.map(_work, pending):
                if code == 0:
                    cleanup_checkpoints(run, policy)
                else:
                    failures += 1
                    print(f"[{run.run_id}] FAILED (exit {code})")
                write_index(manifest, runs)
    finally:
        # Otherwise a raised launch() leaves every completed run's status.json
        # stuck at "running", and --resume would redo them.
        write_index(manifest, runs)

    write_index(manifest, runs)
    done = sum(1 for r in runs if _read_status(r)["state"] == "done")
    print(f"\nDone: {done}/{len(runs)} runs complete, {failures} failed this session.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    parser.add_argument("--resume", action="store_true", help="skip done runs, resume failed ones")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "run the manifest with live teacher models (teachers.mode=live, KD active): "
            "every spec gets the live overrides and results go to runs/<experiment>_live"
        ),
    )
    parser.add_argument(
        "--only", type=str, default=None, help="run only these spec ids (comma-separated)"
    )
    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help=(
            "skip these spec ids (comma-separated), e.g. --skip random_routing until "
            "trihope's action shares exist"
        ),
    )
    parser.add_argument(
        "--max-hours", type=float, default=None, help="stop launching past this budget"
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the config/data validation pass before launching",
    )
    parser.add_argument(
        "--parallel-gpus",
        type=int,
        default=None,
        help=(
            "run N specs CONCURRENTLY, one per GPU (CUDA_VISIBLE_DEVICES pinned). "
            "For a matrix of independent runs this beats DDP: no collectives, no "
            "batch-size constraint, no stream-digest change, N x the runs."
        ),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help=(
            "DDP-shard EACH run across N GPUs via torch.distributed.run. Use for "
            "long single runs (headline), not for the ablation matrix. Requires "
            "data.batch_size (the GLOBAL batch) to be divisible by N."
        ),
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=None,
        help=(
            "run N specs CONCURRENTLY on the ONE visible GPU (no device pinning). "
            "For a small student on a large card: pick N from the peak memory of "
            "a first run (run_summary.json profile.final_peak_mem_gb)."
        ),
    )
    args = parser.parse_args(argv)
    modes = [
        name
        for name, val in (
            ("--parallel-gpus", args.parallel_gpus),
            ("--nproc-per-node", args.nproc_per_node),
            ("--concurrent", args.concurrent),
        )
        if val and val > 1
    ]
    if len(modes) > 1:
        parser.error(
            f"{' and '.join(modes)} are alternative ways to spend the same GPU(s); "
            "pick one."
        )

    manifest = load_manifest(args.manifest)
    if args.live:
        manifest = apply_live(manifest)
    nproc_requested = int(args.nproc_per_node or manifest.get("nproc_per_node", 1))
    runs = expand_matrix(manifest)

    if nproc_requested > 1:
        code = _check_shardable(manifest, nproc_requested)
        if code:
            return code

    if not args.dry_run and not args.no_preflight:
        # One config check before ANY run launches: a broken manifest then
        # fails in about a minute instead of six hours into Group D.
        code = _preflight_manifest(manifest)
        if code:
            return code
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        runs = [r for r in runs if r.spec_id in wanted]
        if not runs:
            print(f"No runs match --only {args.only}")
            return 2
    if args.skip:
        skipped = {x.strip() for x in args.skip.split(",") if x.strip()}
        runs = [r for r in runs if r.spec_id not in skipped]
        if not runs:
            print(f"Every run is excluded by --skip {args.skip}")
            return 2

    if args.dry_run:
        for r in runs:
            print(f"{r.run_id:40s} {_read_status(r)['state']:8s} {' '.join(r.overrides)}")
        return 0

    policy = str(manifest.get("cleanup_checkpoints", "keep_final"))
    nproc = int(args.nproc_per_node or manifest.get("nproc_per_node", 1))
    budget_start = time.time()
    failures = 0

    if args.parallel_gpus and args.parallel_gpus > 1:
        return _run_parallel(runs, manifest, args, policy, nproc=1)
    if args.concurrent and args.concurrent > 1:
        return _run_parallel(runs, manifest, args, policy, nproc=1, share_gpu=True)

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
        code = launch(
            run, manifest["base_config"], resume=resume_this, nproc=nproc
        )
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

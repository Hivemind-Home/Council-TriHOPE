#!/usr/bin/env python
"""Selective rollback of one teacher's permanent writes (task T5.4, E5).

    python scripts/rollback_teacher.py --run runs/<exp>/<spec>-seed<s> \\
        --teacher math_teacher_corrupted --out runs/<exp>/<spec>-seed<s>-rollback

Selective rollback (default): find the first ``consolidation`` event whose
``attribution_share`` for ``--teacher`` is at least ``--min-share``, restore
the ``step_XXXXXXXX_pre_merge`` checkpoint written right before it, and
resume the ORIGINAL run's configuration to the end with
``controller.debug.block_p_for_teachers=[<teacher>]`` — that teacher's P
actions are demoted to F, its dominated adapters are reset on restore, and
everything learned afterwards from the other teachers is kept. The bad
data stays in the stream: this measures containment, not filtering.

``--baseline full_restore``: the only option a full-FT model has. Restore
the ``pre_phase_<phase>`` checkpoint taken before the corrupted phase and
resume with ``train.skip_step_ranges=[[start, end]]`` so the phase is never
trained on — discarding it wholesale, including its clean rows.

Both variants replay the original Hydra overrides (from the run's
``hydra/rank0/.hydra/overrides.yaml``) with the run-dir keys rewritten to
``--out``; the stream config is therefore identical and the digest guard
passes. When the resumed run finishes, ``rollback_summary.json`` in
``--out`` compares its final per-domain losses and retention against the
original run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_DIR_KEYS = (
    "++checkpoint.dir=",
    "++logging.path=",
    "++logging.events_path=",
    "++run.dir=",
    "++hydra.run.dir=",
)


def read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def find_rollback_point(
    events: list[dict], teacher: str, min_share: float = 0.5
) -> Optional[dict]:
    """First merge whose attribution share for ``teacher`` is ≥ ``min_share``.

    Merges without attribution (an empty adapter, or a pre-T5 trace) are
    skipped — nothing to attribute means nothing to revert.
    """
    for e in events:
        if e.get("type") != "consolidation":
            continue
        shares = e.get("attribution_share") or {}
        if float(shares.get(teacher, 0.0)) >= min_share:
            return e
    return None


def find_phase_range(events: list[dict], phase: str) -> Optional[tuple[int, int]]:
    """[first step, last step] of ``phase`` from the phase_start markers."""
    starts = sorted(
        {(int(e["step"]), str(e["phase"])) for e in events if e.get("type") == "phase_start"}
    )
    for i, (start, name) in enumerate(starts):
        if name == phase:
            end = starts[i + 1][0] - 1 if i + 1 < len(starts) else None
            if end is None:
                ends = [int(e["step"]) for e in events if e.get("type") == "phase_end"
                        and e.get("phase") == phase]
                end = max(ends) if ends else start
            return start, end
    return None


def corrupted_phase(events: list[dict]) -> Optional[str]:
    for e in events:
        if e.get("type") == "run_config":
            block = e.get("corrupt_teacher") or {}
            if block.get("enabled"):
                return str(block.get("phase"))
    return None


def original_overrides(run_dir: Path) -> tuple[str, list[str]]:
    """(base config name, Hydra overrides) the original run was launched with."""
    status = json.loads((run_dir / "status.json").read_text()) if (
        run_dir / "status.json"
    ).exists() else {}
    cmd = str(status.get("cmd", ""))
    base_config = "stream_small"
    parts = cmd.split()
    if "--config-name" in parts:
        base_config = parts[parts.index("--config-name") + 1]
    candidates = sorted((run_dir / "hydra").glob("rank*/.hydra/overrides.yaml")) + sorted(
        (run_dir / "hydra").glob(".hydra/overrides.yaml")
    )
    overrides: list[str] = []
    if candidates:
        overrides = [str(o) for o in (yaml.safe_load(candidates[0].read_text()) or [])]
    elif parts:
        overrides = [p for p in parts[parts.index("--config-name") + 2:]] if (
            "--config-name" in parts
        ) else []
    return base_config, overrides


def rewrite_overrides(overrides: list[str], out_dir: Path, extra: list[str]) -> list[str]:
    """Drop the original run-dir keys, add ours, append ``extra``."""
    kept = [o for o in overrides if not o.startswith(_RUN_DIR_KEYS)
            and not o.startswith("++checkpoint.resume_from=")
            and not o.startswith("checkpoint.resume_from=")]
    ours = [
        f"++checkpoint.dir={out_dir / 'checkpoints'}",
        f"++logging.path={out_dir / 'metrics.jsonl'}",
        f"++logging.events_path={out_dir / 'events.jsonl'}",
        f"++run.dir={out_dir}",
        f"++hydra.run.dir={out_dir / 'hydra'}/rank${{oc.env:RANK,0}}",
    ]
    return kept + ours + list(extra)


def plan(
    run_dir: Path,
    teacher: str,
    out_dir: Path,
    *,
    min_share: float = 0.5,
    baseline: Optional[str] = None,
) -> dict[str, Any]:
    events = read_events(run_dir)
    base_config, overrides = original_overrides(run_dir)
    ckpt_root = run_dir / "checkpoints"
    if baseline == "full_restore":
        phase = corrupted_phase(events)
        if phase is None:
            raise SystemExit("full_restore: the run has no enabled corrupt_teacher block")
        rng = find_phase_range(events, phase)
        if rng is None:
            raise SystemExit(f"full_restore: phase {phase!r} not found in the trace")
        start, end = rng
        ckpt = ckpt_root / f"step_{start - 1:08d}_pre_phase_{phase}"
        extra = [
            f"++checkpoint.resume_from={ckpt}",
            f"++train.skip_step_ranges=[[{start},{end}]]",
        ]
        info: dict[str, Any] = {"mode": "full_restore", "phase": phase,
                                "skip_range": [start, end], "restore_step": start - 1}
    else:
        point = find_rollback_point(events, teacher, min_share)
        if point is None:
            raise SystemExit(
                f"no consolidation event attributes ≥ {min_share:.2f} of a merge to "
                f"{teacher!r}; nothing to roll back (containment held)."
            )
        step = int(point["step"])
        ckpt = ckpt_root / f"step_{step:08d}_pre_merge"
        extra = [
            f"++checkpoint.resume_from={ckpt}",
            f"++controller.debug.block_p_for_teachers=[{teacher}]",
            # the same threshold decides which pending adapters are reset
            f"++controller.debug.block_min_share={min_share}",
        ]
        info = {"mode": "selective", "restore_step": step, "module": point.get("module"),
                "attribution_share": point.get("attribution_share")}
    if not ckpt.exists():
        raise SystemExit(
            f"checkpoint {ckpt} does not exist. The original run needs "
            "controller.consolidation.checkpoint_before_merge=true and checkpoint.keep_tagged=0 "
            "(and checkpoint.save_before_phases for full_restore)."
        )
    cmd = [sys.executable, str(REPO_ROOT / "train.py"), "--config-name", base_config,
           *rewrite_overrides(overrides, out_dir, extra)]
    info.update({"teacher": teacher, "checkpoint": str(ckpt), "cmd": cmd,
                 "original_run": str(run_dir), "out": str(out_dir)})
    return info


def _final_losses(summary: dict) -> tuple[dict, Optional[float]]:
    history = (summary.get("retention") or {}).get("history") or []
    final = history[-1].get("loss", {}) if history else {}
    worst = None
    for entry in history:
        for _d, delta in (entry.get("retention_delta") or {}).items():
            worst = delta if worst is None else max(worst, delta)
    return dict(final), worst


def write_summary(info: dict, run_dir: Path, out_dir: Path) -> Path:
    orig = json.loads((run_dir / "run_summary.json").read_text()) if (
        run_dir / "run_summary.json"
    ).exists() else {}
    new = json.loads((out_dir / "run_summary.json").read_text()) if (
        out_dir / "run_summary.json"
    ).exists() else {}
    o_final, o_worst = _final_losses(orig)
    n_final, n_worst = _final_losses(new)
    payload = {
        **{k: v for k, v in info.items() if k != "cmd"},
        "original": {"final_loss": o_final, "worst_retention_delta": o_worst,
                     "permanent_writes": orig.get("permanent_writes")},
        "rollback": {"final_loss": n_final, "worst_retention_delta": n_worst,
                     "permanent_writes": new.get("permanent_writes")},
        "final_loss_delta": {d: n_final[d] - o_final[d] for d in n_final if d in o_final},
    }
    path = out_dir / "rollback_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-share", type=float, default=0.5)
    parser.add_argument("--baseline", choices=["full_restore"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = parser.parse_args(argv)

    info = plan(args.run, args.teacher, args.out, min_share=args.min_share,
                baseline=args.baseline)
    print(json.dumps({k: v for k, v in info.items() if k != "cmd"}, indent=2))
    print(" ".join(info["cmd"]))
    if args.dry_run:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rollback_plan.json").write_text(json.dumps(info, indent=2))
    code = subprocess.run(info["cmd"], cwd=str(REPO_ROOT)).returncode
    if code:
        print(f"resumed run failed with exit {code}")
        return code
    path = write_summary(info, args.run, args.out)
    print(f"rollback summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

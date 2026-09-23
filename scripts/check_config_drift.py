#!/usr/bin/env python
"""Verify that new runs resolve to the config the committed live runs used.

The live thresholds were a box-local edit (CAMPAIGN_NOTES §10) and the live
code commit was not recorded, so every run of the final batch is checked
against a committed reference before it may be pooled with the 115 live runs.

For each run dir, the resolved ``hydra/rank0/.hydra/config.yaml`` is diffed
against a reference run's, ignoring seed and output paths. A difference is
*allowed* only on a key that the new run's own overrides, or the reference's,
set explicitly. Anything else is drift (a changed default, a stale checkout):
the script prints it and exits 1.

Reference, per run (``--ref`` overrides it):
  * a spec that exists in the committed live campaign (e.g. molf_style_a0p7,
    no_cosine): that spec's committed seed-1337 run -> must match except seed;
  * anything else (trihope_sentinel, frozen_blocks, periodic_merge, pgate_*):
    the committed live E1 trihope at the same seed.

    python scripts/check_config_drift.py runs/priority_s3_v1_live/*-seed*
    python scripts/check_config_drift.py runs/priority_s2_v1_live/*-seed* --committed results_live
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

IGNORE_EXACT = {
    "train.seed",
    "checkpoint.dir",
    "logging.path",
    "logging.events_path",
    "run.dir",
}
IGNORE_PREFIX = ("hydra.",)


def flatten(d: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def override_keys(overrides_yaml: Path) -> set[str]:
    if not overrides_yaml.exists():
        return set()
    keys = set()
    for item in yaml.safe_load(overrides_yaml.read_text()) or []:
        m = re.match(r"^[+~]*([^=]+)=", str(item))
        if m:
            keys.add(m.group(1).strip())
    return keys


def hydra_dir(run: Path) -> Path:
    for cand in (run / "hydra" / "rank0" / ".hydra", run / "hydra" / ".hydra"):
        if (cand / "config.yaml").exists():
            return cand
    raise FileNotFoundError(f"no resolved config under {run}/hydra")


def split_run_name(run: Path) -> tuple[str, int]:
    m = re.match(r"^(.*)-seed(\d+)$", run.name)
    if not m:
        raise ValueError(f"run dir name must end in -seed<N>: {run}")
    return m.group(1), int(m.group(2))


def find_reference(run: Path, committed: Path) -> Path:
    spec, seed = split_run_name(run)
    # Only live-mode matrices are references (on the box, runs/ also holds the
    # cache-mode dirs of the same spec ids, which used other thresholds).
    same_spec = sorted(committed.glob(f"*_live/{spec}-seed1337"))
    same_spec = [p for p in same_spec if p.resolve() != run.resolve()]
    if spec != "trihope_sentinel" and same_spec:
        return same_spec[0]
    ref = committed / "baselines_small_v1_live" / f"trihope-seed{seed}"
    if not ref.exists():
        raise FileNotFoundError(f"reference {ref} missing")
    return ref


def check(run: Path, ref: Path) -> list[str]:
    new_dir, ref_dir = hydra_dir(run), hydra_dir(ref)
    new = flatten(yaml.safe_load((new_dir / "config.yaml").read_text()))
    old = flatten(yaml.safe_load((ref_dir / "config.yaml").read_text()))
    allowed = override_keys(new_dir / "overrides.yaml") | override_keys(
        ref_dir / "overrides.yaml"
    )
    problems = []
    for key in sorted(set(new) | set(old)):
        if key in IGNORE_EXACT or key.startswith(IGNORE_PREFIX):
            continue
        a, b = new.get(key, "<absent>"), old.get(key, "<absent>")
        if a == b:
            continue
        tag = "allowed" if key in allowed else "DRIFT"
        problems.append(f"  [{tag}] {key}: ref={b!r} new={a!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument(
        "--committed",
        type=Path,
        default=Path("results_live"),
        help="dir holding the committed live runs (results_live, or runs/ on the box)",
    )
    ap.add_argument("--ref", type=Path, default=None, help="force one reference run dir")
    args = ap.parse_args(argv)

    drift = False
    for run in args.runs:
        if not run.is_dir():
            continue
        try:
            ref = args.ref or find_reference(run, args.committed)
            lines = check(run, ref)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{run}: SKIP ({exc})")
            continue
        bad = [ln for ln in lines if "[DRIFT]" in ln]
        drift |= bool(bad)
        status = "DRIFT" if bad else "ok"
        print(f"{run} vs {ref}: {status} ({len(lines) - len(bad)} allowed diffs)")
        for ln in lines:
            print(ln)
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())

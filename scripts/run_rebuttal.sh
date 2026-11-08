#!/usr/bin/env bash
# ICLR-2027 rebuttal batch: one script for the GPU operator.
#
#   bash scripts/run_rebuttal.sh check     # env + tests + preflight + dry-run (10 min, no GPU work)
#   bash scripts/run_rebuttal.sh smoke     # 7 tiny CPU runs of every arm, ~20 min, verifies the new code paths
#   bash scripts/run_rebuttal.sh launch    # the 21 live runs, 3 at a time on the GPU (resumable; re-run to continue)
#   bash scripts/run_rebuttal.sh early     # (second shell, ~1 h in) loss + commits so far per run
#   bash scripts/run_rebuttal.sh post      # drift audit, commit schedules, report, results tarball
#   bash scripts/run_rebuttal.sh all       # check + smoke + launch + post
#
# Read docs/REBUTTAL_RUNBOOK.md first. Every stage is idempotent: `launch`
# skips finished runs and resumes interrupted ones from their last checkpoint.
#
# Knobs (environment variables):
#   CONCURRENT=3      runs sharing the GPU (3 on an A6000/A100-80G; 2 on a 40 GB card)
#   PY=python         interpreter (the repo's venv)
#   MANIFEST=configs/experiments/rebuttal_v1.yaml
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-$(command -v python || command -v python3)}"   # the repo's venv; set PY=... to be explicit
CONCURRENT="${CONCURRENT:-3}"
MANIFEST="${MANIFEST:-configs/experiments/rebuttal_v1.yaml}"
EXP="$($PY - "$MANIFEST" <<'EOF'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
print(m["experiment"] + "_live")
EOF
)"
RUNS="runs/$EXP"
STAGE="${1:-all}"

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; }

stage_check() {
  log "branch / commit"
  git rev-parse --abbrev-ref HEAD; git log -1 --format='%h %ci %s'
  if [ -n "$(git status --porcelain -- src scripts configs tests)" ]; then
    echo "WARNING: uncommitted changes under src/ scripts/ configs/ tests/ — the provenance will not match the branch."
    git status --short -- src scripts configs tests
  fi
  log "python / torch / GPU"
  $PY - <<'EOF'
import torch, hivemind
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
  log "unit tests for the new code paths (CPU, ~3 min)"
  $PY -m pytest tests/test_rebuttal_controls.py tests/test_policy.py tests/test_debug_controls.py \
      tests/test_baseline_policies.py tests/test_evaluation.py tests/test_training_loop.py -q
  log "preflight (metadata only) + dry-run of the matrix"
  $PY -m hivemind preflight --config-name stream_small --metadata-only
  $PY scripts/run_experiment.py "$MANIFEST" --live --dry-run | tail -n 25
}

stage_smoke() {
  log "CPU smoke of every arm on stream_smoke (70 steps each)"
  $PY scripts/run_experiment.py configs/experiments/rebuttal_smoke.yaml --resume
  log "smoke checks"
  $PY - <<'EOF'
import json, pathlib, sys
root = pathlib.Path("runs/rebuttal_smoke")
bad = 0
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    s = d / "run_summary.json"
    if not s.exists():
        print(f"{d.name}: NO run_summary.json"); bad += 1; continue
    j = json.loads(s.read_text()); ex = j.get("extra", {})
    ok = (d / "commits.jsonl").exists() and (d / "eval_rows.jsonl").exists()
    line = f"{d.name:<36} trainable={str(ex.get('trainable')):<7} commits.jsonl={ (d/'commits.jsonl').exists() } eval_rows={ (d/'eval_rows.jsonl').exists() } p_count={j['ledger_totals'].get('p_count')}"
    if d.name.startswith("shared_frozen"):
        chk = ex.get("shared_frozen_check") or {}
        line += f" shared_unchanged={chk.get('unchanged')} ({chk.get('coords')} coords)"
        ok = ok and chk.get("unchanged") is True
    print(line + ("" if ok else "   <-- FAIL")); bad += 0 if ok else 1
sys.exit(1 if bad else 0)
EOF
  $PY scripts/commit_schedule.py runs/rebuttal_smoke/*-seed1337 --reference runs/rebuttal_smoke/trihope-seed1337 --early-window 10
}

stage_launch() {
  log "launching $MANIFEST --live --resume --concurrent $CONCURRENT  ->  $RUNS"
  mkdir -p "$RUNS"
  { git rev-parse HEAD; git status --short; $PY -c 'import torch;print(torch.__version__)'; } > "$RUNS/provenance_$(date -u +%Y%m%dT%H%M%SZ).txt"
  # The runner exits non-zero if ANY run failed; `post` can still run on the rest.
  $PY scripts/run_experiment.py "$MANIFEST" --live --resume --concurrent "$CONCURRENT" || \
    echo "run_experiment exited non-zero: some run failed. Re-run 'launch' to resume it; see $RUNS/*/status.json"
}

stage_early() {
  # Run this ~30-60 min after `launch`: are the first runs learning and committing?
  # In particular shared_frozen_trihope must show a falling loss and, by the end
  # of code_recurrent (step 1999), commits > 0. (On the CPU smoke the frozen arms
  # never commit: that student is randomly initialised, so freezing its
  # embeddings leaves nothing to learn with. The pretrained 0.6B is different.)
  log "early check: $RUNS"
  for d in "$RUNS"/*-seed*; do
    [ -f "$d/metrics.jsonl" ] || continue
    $PY - "$d" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in (d/"metrics.jsonl").read_text().splitlines() if l.strip()]
if not rows: print(f"{d.name:<40} no metrics yet"); sys.exit()
first, last = rows[0], rows[-1]
commits = sum(1 for l in (d/"commits.jsonl").read_text().splitlines() if '"p": 0' not in l) if (d/"commits.jsonl").exists() else "-"
print(f"{d.name:<40} step {last['step']:>5}  loss {first.get('loss/total',0):.3f} -> {last.get('loss/total',0):.3f}"
      f"  committing steps so far: {commits}")
EOF
  done
}

stage_post() {
  log "status"
  $PY - "$RUNS" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    st = d / "status.json"
    print(f"{d.name:<40} {json.loads(st.read_text()).get('state') if st.exists() else 'missing'}")
EOF
  log "config drift: each run's resolved config vs the committed live E1 trihope (only the arm's own overrides may differ)"
  if [ -d results_live ]; then
    $PY scripts/check_config_drift.py "$RUNS"/*-seed* --committed results_live || \
      echo "CONFIG DRIFT reported above — send this output with the results"
  fi
  log "weight drift vs the pretrained model (writes weight_drift.json in each run)"
  shopt -s nullglob
  frozen=( "$RUNS"/shared_frozen_*-seed* )
  others=( "$RUNS"/anti_commit-seed* "$RUNS"/yoked_random-seed* "$RUNS"/trihope_e256-seed* )
  if [ ${#frozen[@]} -gt 0 ]; then
    $PY scripts/weight_drift.py "${frozen[@]}" --expect-shared-unchanged || echo "DRIFT CHECK FAILED on a shared_frozen arm — report this, do not hide it"
  fi
  if [ ${#others[@]} -gt 0 ]; then $PY scripts/weight_drift.py "${others[@]}" || true; fi
  log "commit schedules (timing + volume + overlap), vs trihope_e256 of the same seed"
  mkdir -p "$RUNS/analysis"
  for seed in 1337 2024 7; do
    ref="$RUNS/trihope_e256-seed$seed"
    arms=( "$RUNS"/*-seed$seed )
    [ -d "$ref" ] || { echo "no reference for seed $seed yet"; continue; }
    $PY scripts/commit_schedule.py "${arms[@]}" --reference "$ref" --json "$RUNS/analysis/commit_schedule_seed$seed.json" || true
  done
  log "aggregate report"
  $PY -m analysis.run_report "$RUNS" || true
  log "tarball of what the paper needs (no checkpoints)"
  tar --exclude='checkpoints' --exclude='torchrun_logs' -czf "rebuttal_results_$(date -u +%Y%m%dT%H%MZ).tgz" "$RUNS"
  ls -la rebuttal_results_*.tgz | tail -1
  echo "Send the tarball (or commit $RUNS without checkpoints to a results branch)."
}

case "$STAGE" in
  check)  stage_check ;;
  smoke)  stage_smoke ;;
  launch) stage_launch ;;
  early)  stage_early ;;
  post)   stage_post ;;
  all)    stage_check; stage_smoke; stage_launch; stage_post ;;
  *) echo "usage: $0 {check|smoke|launch|early|post|all}"; exit 2 ;;
esac

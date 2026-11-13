#!/usr/bin/env bash
# ICLR-2027 rebuttal, GPU batches of 2026-11-13: one script for the operator.
# Read docs/GPU_RUN_NOV13.md first.
#
#   bash scripts/run_rebuttal_nov13.sh check   # ~5 min: branch, GPU, references present, tests, dry-runs
#   bash scripts/run_rebuttal_nov13.sh run     # the 20 GPU runs, in priority order (resumable: re-run to continue)
#   bash scripts/run_rebuttal_nov13.sh smoke   # (second shell, CPU, ~15 min) oracle arms on the tiny stream
#   bash scripts/run_rebuttal_nov13.sh early   # (second shell, ~30-45 min after `run`) progress, OOM, oracle scoring
#   bash scripts/run_rebuttal_nov13.sh post    # checks + tables + tarball (fine on a partial batch)
#
# `run` order (each step waits for the previous one):
#   1. oracle bracket, shared-frozen regime   6 runs   <- the rebuttal needs these first
#   2. extra seeds 42/314, shared-frozen      8 runs   (stage B starts only once both
#                                                       stage-A references are DONE)
#   3. oracle bracket, shared-trainable       6 runs
# About 1.6-1.8 h per run with 3 sharing the GPU (measured on batch 1): step 1 ~3.5-4 h,
# step 2 ~5 h, step 3 ~3.5-4 h.
#
# Knobs: CONCURRENT=3 (runs sharing the GPU; use 2 after any CUDA OOM), PY=python
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-$(command -v python || command -v python3)}"
CONCURRENT="${CONCURRENT:-3}"
STAGE="${1:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS="runs/rebuttal_nov13_logs"; mkdir -p "$LOGS"
ANA="runs/rebuttal_nov13_analysis"

ORACLE=configs/experiments/rebuttal_oracle.yaml
SEEDS_A=configs/experiments/rebuttal_seeds_ref.yaml
SEEDS_B=configs/experiments/rebuttal_seeds_sched.yaml
V1=runs/rebuttal_v1_live
VOL=runs/rebuttal_volume_live
ORA=runs/rebuttal_oracle_live
SA=runs/rebuttal_seeds_live
SB=runs/rebuttal_seeds_sched_live
FROZEN_ORACLE=shared_frozen_schedule_oracle_bad,shared_frozen_schedule_oracle_good
TRAIN_ORACLE=schedule_oracle_bad,schedule_oracle_good

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; }
state() { [ -f "$1/status.json" ] && "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('state'))" "$1/status.json" || echo missing; }

stage_check() {
  local bad=0
  log "branch / commit (must be rebuttal/extra-seeds)"
  git rev-parse --abbrev-ref HEAD; git log -1 --format='%h %ci %s'
  if [ -n "$(git status --porcelain -- src scripts configs tests)" ]; then
    echo "WARNING: uncommitted changes under src/ scripts/ configs/ tests/:"; git status --short -- src scripts configs tests
  fi
  log "python / torch / GPU ($PY)"
  if ! "$PY" -c "import numpy, torch, hivemind" 2>/dev/null; then
    echo "This interpreter cannot import numpy/torch/hivemind. Activate the repo's venv, or run with"
    echo "PY=/path/to/venv/bin/python bash scripts/run_rebuttal_nov13.sh <stage>  (same PY for every stage)"
    return 1
  fi
  "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  echo "(batch 1 peaked at ~29 GB per run; 3 concurrent need ~90 GB. Anything else on the GPU -> CONCURRENT=2)"
  log "repair: runs interrupted under the old runner were wrongly marked 'done' -> mark 'failed' so 'run' finishes them"
  "$PY" - <<'EOF'
import json, pathlib
n = 0
for st in pathlib.Path("runs").glob("rebuttal_*_live/*-seed*/status.json"):
    summ = st.parent / "run_summary.json"
    s = json.loads(st.read_text())
    if s.get("state") == "done" and summ.exists() and json.loads(summ.read_text()).get("interrupted"):
        s.update(state="failed", interrupted=True, repaired="2026-11-13 runner fix")
        st.write_text(json.dumps(s, indent=2)); n += 1
        print(f"  repaired (will be resumed): {st.parent}")
print(f"  {n} run(s) repaired")
EOF
  log "batch-1 references the oracle arms read (must all be 'done')"
  for r in shared_frozen_trihope trihope_e256; do for s in 1337 2024 7; do
    d="$V1/$r-seed$s"; st=$(state "$d")
    if [ "$st" = done ] && [ -s "$d/commits.jsonl" ]; then echo "ok   $d"; else echo "MISSING/NOT DONE ($st)  $d"; bad=1; fi
  done; done
  log "unit tests for the new code paths (CPU, ~3-5 min)"
  "$PY" -m pytest tests/test_rebuttal_controls.py tests/test_policy.py tests/test_masked_adamw.py \
      tests/test_training_loop.py tests/test_evaluation.py -q -p no:cacheprovider || bad=1
  log "preflight + dry-runs"
  "$PY" -m hivemind preflight --config-name stream_small --metadata-only || bad=1
  for m in "$ORACLE" "$SEEDS_A" "$SEEDS_B"; do
    "$PY" scripts/run_experiment.py "$m" --live --dry-run | awk '{print "  ",$1,$2}' | grep -- -seed || bad=1
  done
  if [ $bad -ne 0 ]; then echo; echo "CHECK FAILED - send this output, do not launch"; return 1; fi
  echo; echo "CHECK OK - launch with: bash scripts/run_rebuttal_nov13.sh run"
}

refs_done() {  # both stage-A references finished?
  [ "$(state "$SA/shared_frozen_trihope-seed42")" = done ] && [ "$(state "$SA/shared_frozen_trihope-seed314")" = done ]
}

stage_run() {
  { git rev-parse HEAD; git status --short; "$PY" -c 'import torch;print(torch.__version__)'; } > "$LOGS/provenance_$TS.txt"

  log "STEP 1/3: oracle bracket, shared-frozen (6 runs, ~3.5-4 h)"
  "$PY" scripts/run_experiment.py "$ORACLE" --only "$FROZEN_ORACLE" --live --resume --concurrent "$CONCURRENT" \
    || echo "step 1: some run failed (see $ORA/*/status.json); continuing; re-run 'run' later to retry it"

  log "STEP 2/3: extra seeds 42/314 (stage A refs; stage B as soon as both refs are done)"
  "$PY" scripts/run_experiment.py "$SEEDS_A" --live --resume --concurrent "$CONCURRENT" &
  local pa=$!
  while kill -0 "$pa" 2>/dev/null && ! refs_done; do sleep 60; done
  if refs_done; then
    local cb=$(( CONCURRENT > 1 ? CONCURRENT - 1 : 1 ))
    kill -0 "$pa" 2>/dev/null || cb="$CONCURRENT"
    log "stage-A references done -> stage B (--concurrent $cb)"
    "$PY" scripts/run_experiment.py "$SEEDS_B" --live --resume --concurrent "$cb" \
      || echo "stage B: some run failed; re-run 'run' later to retry it"
  else
    echo "stage A ended but a reference is not done -> stage B NOT started (it would read a partial schedule)."
    echo "Re-run 'run' to resume stage A; stage B follows automatically."
  fi
  wait "$pa" || echo "stage A: some run failed; re-run 'run' later to retry it"

  log "STEP 3/3: oracle bracket, shared-trainable (6 runs, ~3.5-4 h)"
  "$PY" scripts/run_experiment.py "$ORACLE" --only "$TRAIN_ORACLE" --live --resume --concurrent "$CONCURRENT" \
    || echo "step 3: some run failed; re-run 'run' later to retry it"
  log "run finished -> bash scripts/run_rebuttal_nov13.sh post"
}

stage_smoke() {
  # CPU only, limited threads so it does not slow the GPU runs' data loading.
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
  if [ "$(state runs/rebuttal_smoke/trihope-seed1337)" != done ]; then
    log "smoke reference missing -> running it first (~5 min)"
    "$PY" scripts/run_experiment.py configs/experiments/rebuttal_smoke.yaml --only trihope --resume || return 1
  fi
  log "oracle arms on stream_smoke (70 steps each, CPU)"
  "$PY" scripts/run_experiment.py configs/experiments/rebuttal_oracle_smoke.yaml --resume --concurrent 2 || return 1
  log "smoke checks (both must pass)"
  "$PY" scripts/oracle_check.py runs/rebuttal_oracle_smoke/*-seed1337 --min-side 0.9
  "$PY" scripts/commit_schedule.py runs/rebuttal_oracle_smoke/*-seed1337 \
      --reference runs/rebuttal_smoke/trihope-seed1337 --early-window 10 --expect-match | grep -E "==|TOTAL|FAIL"
}

stage_early() {
  log "progress (step, loss first -> now, committing steps, OOM)"
  shopt -s nullglob
  for d in "$ORA"/*-seed* "$SA"/*-seed* "$SB"/*-seed*; do
    [ -f "$d/metrics.jsonl" ] || { echo "$(basename "$d"): $(state "$d"), no metrics yet"; continue; }
    "$PY" - "$d" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = [r for r in (json.loads(l) for l in (d/"metrics.jsonl").read_text().splitlines() if l.strip())
        if "loss/total" in r]  # eval/summary rows carry no training loss
st = json.loads((d/"status.json").read_text()).get("state") if (d/"status.json").exists() else "?"
c = d/"commits.jsonl"
commits = sum(1 for l in c.read_text().splitlines() if l.strip() and '"p": 0' not in l) if c.exists() else 0
log = (d/"stdout.log").read_text(errors="ignore") if (d/"stdout.log").exists() else ""
oom = "  <-- CUDA OOM: stop, re-run with CONCURRENT=2" if "out of memory" in log.lower() else ""
if rows:
    print(f"{d.name:<44} {st:<8} step {rows[-1]['step']:>5}  loss {rows[0].get('loss/total',0):.3f} -> "
          f"{rows[-1].get('loss/total',0):.3f}  committing steps {commits}{oom}")
EOF
  done
  local o=( "$ORA"/*oracle*-seed* )
  if [ ${#o[@]} -gt 0 ]; then
    log "oracle scoring so far (all fallback before step 500 is expected: no earlier domain yet)"
    "$PY" scripts/oracle_check.py "${o[@]}" --min-side 0.9 || \
      echo "(FAIL is expected only while 'scored' is 0; FAIL with scored > 10 -> stop and report)"
  fi
}

stage_post() {
  mkdir -p "$ANA"; shopt -s nullglob
  log "status"
  for d in "$ORA"/*-seed* "$SA"/*-seed* "$SB"/*-seed*; do printf '%-48s %s\n' "$(basename "$d")" "$(state "$d")"; done | tee "$ANA/status.txt"
  log "1. shared parameters unchanged in every frozen run (must say ok)"
  local fz=( "$ORA"/shared_frozen_*-seed* "$SA"/*-seed* "$SB"/*-seed* )
  [ ${#fz[@]} -gt 0 ] && { "$PY" scripts/weight_drift.py "${fz[@]}" --expect-shared-unchanged 2>&1 | tee "$ANA/weight_drift.txt" \
      || echo "DRIFT CHECK FAILED - report it, do not re-run"; }
  log "2. oracle manipulation check (must PASS)"
  local o=( "$ORA"/*oracle*-seed* )
  [ ${#o[@]} -gt 0 ] && "$PY" scripts/oracle_check.py "${o[@]}" 2>&1 | tee "$ANA/oracle_check.txt"
  log "3. commit volume = reference (ratio 1.00)"
  {
    for s in 1337 2024 7; do
      a=( "$ORA"/shared_frozen_*-seed$s ); [ ${#a[@]} -gt 0 ] && "$PY" scripts/commit_schedule.py "${a[@]}" --reference "$V1/shared_frozen_trihope-seed$s" --expect-match
      a=( "$ORA"/schedule_*-seed$s );      [ ${#a[@]} -gt 0 ] && "$PY" scripts/commit_schedule.py "${a[@]}" --reference "$V1/trihope_e256-seed$s" --expect-match
    done
    for s in 42 314; do
      a=( "$SB"/*-seed$s ); [ ${#a[@]} -gt 0 ] && "$PY" scripts/commit_schedule.py "${a[@]}" --reference "$SA/shared_frozen_trihope-seed$s" --expect-match
    done
  } 2>&1 | tee "$ANA/commit_schedule.txt" | grep -E "^== |TOTAL|FAIL"
  log "4. M1 table, all batches, all seeds"
  "$PY" scripts/m1_rows.py "$V1" "$VOL" "$ORA" "$SA" "$SB" 2>&1 | tee "$ANA/m1_rows.txt"
  log "5. tarball (no checkpoints)"
  local tgz="rebuttal_nov13_results_$TS.tgz"
  tar --exclude=checkpoints --exclude=torchrun_logs --exclude='events.jsonl' -czf "$tgz" \
      $(ls -d "$ORA" "$SA" "$SB" "$ANA" "$LOGS" 2>/dev/null)
  ls -la "$tgz"; echo "Send $tgz (and say which steps of 'run' had finished)."
}

case "$STAGE" in
  check) stage_check 2>&1 | tee "$LOGS/check_$TS.log" ;;
  run)   stage_run   2>&1 | tee "$LOGS/run_$TS.log" ;;
  smoke) stage_smoke 2>&1 | tee "$LOGS/smoke_$TS.log" ;;
  early) stage_early 2>&1 | tee "$LOGS/early_$TS.log" ;;
  post)  stage_post  2>&1 | tee "$LOGS/post_$TS.log" ;;
  *) echo "usage: $0 {check|run|smoke|early|post}"; exit 2 ;;
esac

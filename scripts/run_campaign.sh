#!/usr/bin/env bash
# The ICLR-2027 campaign as four stages with the two human stops built in.
#
#   scripts/run_campaign.sh gate     # trihope × 3 seeds → report → signal diagnostic. STOP: threshold decision.
#   scripts/run_campaign.sh e1       # rest of E1 (random_routing last) → report.       STOP: go/no-go on Figure 1.
#   scripts/run_campaign.sh e5       # bad-teacher runs + all rollbacks → report.
#   scripts/run_campaign.sh tier3    # E2, E3, E4, ablations → reports incl. runs/figure1.
#   scripts/run_campaign.sh all      # gate → e1 → e5 → tier3 without stopping (only if the decisions are already made).
#
# Every stage is idempotent: the runner's --resume skips finished runs and
# resumes interrupted ones bit-exactly, so rerunning a stage after a crash or
# Ctrl+C is always safe. CONCURRENT (default 5) is how many runs share the GPU;
# DRY_RUN=1 prints the matrices and launches nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-}"
CONCURRENT="${CONCURRENT:-5}"
RUN=(python scripts/run_experiment.py)
[ "${DRY_RUN:-0}" = "1" ] && RUN+=(--dry-run)
REPORT() { [ "${DRY_RUN:-0}" = "1" ] && return 0; python -m analysis.run_report "$@"; }

stage_gate() {
  echo "== gate: trihope, all seeds"
  "${RUN[@]}" configs/experiments/baselines_small.yaml --only trihope --resume
  REPORT runs/baselines_small_v1
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  echo "== gate: signal diagnostic (read the novel_inject block before launching e1)"
  python scripts/diagnose_p.py runs/baselines_small_v1/trihope-seed1337 | tee runs/baselines_small_v1/diagnose_trihope-seed1337.txt
  if grep -q "C     p10 0.000  p50 0.000  p90 0.000  max 0.000" runs/baselines_small_v1/diagnose_trihope-seed1337.txt; then
    echo "!! C-bar is dead (0.000 everywhere): this checkout predates the tracked-moments fix. Stop." >&2
    exit 3
  fi
  cat <<'EOF'

STOP. Decide the E1 thresholds now (docs/OPERATOR_GUIDE.md §3): if the
novel_inject R share is well under half, set controller.policy.surprise_high /
repetition_low in configs/stream_small.yaml, delete runs/baselines_small_v1,
and rerun `gate`. Otherwise continue with `e1`.
EOF
}

stage_e1() {
  echo "== e1: every spec except random_routing"
  "${RUN[@]}" configs/experiments/baselines_small.yaml --resume --skip random_routing --concurrent "$CONCURRENT"
  REPORT runs/baselines_small_v1                      # writes the action-share file random_routing reads
  echo "== e1: random_routing (needs the shares file above)"
  "${RUN[@]}" configs/experiments/baselines_small.yaml --resume --only random_routing --concurrent 3
  REPORT runs/baselines_small_v1
  cat <<'EOF'

STOP. Go/no-go (docs/TriHOPE_ICLR2027_Reframe.md §9) on
runs/baselines_small_v1/analysis/pareto_budget.png and budget_curve.csv.
Write runs/baselines_small_v1/RESULTS.md and fill docs/STATUS.md §2.
EOF
}

stage_e5() {
  echo "== e5: corrupted-teacher runs"
  "${RUN[@]}" configs/experiments/bad_teacher_small.yaml --resume --concurrent "$CONCURRENT"
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  echo "== e5: selective rollback vs full restore"
  for seed in 1337 2024 7; do
    r=runs/bad_teacher_small_v1/trihope-seed$seed
    [ -d "$r-rollback" ]    || python scripts/rollback_teacher.py --run "$r" --teacher math_teacher_corrupted --out "$r-rollback"
    [ -d "$r-fullrestore" ] || python scripts/rollback_teacher.py --run "$r" --teacher math_teacher_corrupted --baseline full_restore --out "$r-fullrestore"
    g=runs/bad_teacher_small_v1/gradient_routing-seed$seed
    [ -d "$g-fullrestore" ] || python scripts/rollback_teacher.py --run "$g" --teacher math_teacher_corrupted --baseline full_restore --out "$g-fullrestore"
  done
  REPORT runs/bad_teacher_small_v1
}

stage_tier3() {
  for m in r_tier_small budget_sweep_small p_study_small ablation_grid; do
    echo "== tier3: $m"
    "${RUN[@]}" configs/experiments/$m.yaml --resume --concurrent "$CONCURRENT"
  done
  REPORT runs/baselines_small_v1 runs/budget_sweep_small_v1 --out runs/figure1
  REPORT runs/r_tier_small_v1
  REPORT runs/p_study_small_v1
  REPORT runs/baselines_small_v1 runs/ablation_grid_v1 --out runs/ablations   # deltas vs E1's trihope
}

case "$STAGE" in
  gate)  stage_gate ;;
  e1)    stage_e1 ;;
  e5)    stage_e5 ;;
  tier3) stage_tier3 ;;
  all)   stage_gate; stage_e1; stage_e5; stage_tier3 ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac

# GPU runbook — the ICLR-2027 campaign on one 96 GB GPU

> Step-by-step version with health signals and troubleshooting:
> [docs/RUN_PIPELINE_2026-09-04.md](RUN_PIPELINE_2026-09-04.md).

Every command below runs from the repository root on the GPU machine.
Cache-mode teachers mean **no teacher model is ever loaded**: the only model
in memory is the Qwen3-0.6B student, so one 96 GB card can run several
`stream_small` runs side by side (`--concurrent N`, step 2).

`stream_small` (Qwen3-0.6B, 6000 steps, ~1.25 h/run sequential on an
A100/L40S; +30 % for replay specs) is the experiment config for E1–E5.
`stream_headline` (1.7B, 12000 steps, ~6 h/run) is an optional final tier.

## 0. Once: environment, tests, smoke

```bash
pip install -e ".[dev,data,analysis]"
python -m pytest tests/ -q                                # all green, 2 GPU skips become real tests
python train.py --config-name stream_smoke                 # ~2 min on GPU, real math+medical data
python train.py --config-name stream_smoke ++controller.retrieval.replay_on_hit=true   # expect write/replay_count > 0
python train.py --config-name stream_smoke ++controller.retrieval.replay_on_hit=true \
    checkpoint.resume_from=50                              # final loss identical to the run above
python -m hivemind preflight --config-name stream_small --metadata-only
for m in baselines_small r_tier_small budget_sweep_small p_study_small ablation_grid bad_teacher_small; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --dry-run   # runs preflight once per manifest
done
```

Then read `runs/.../run_summary.json → profile.final_peak_mem_gb` from the
first real run (step 1 below) and pick `N = floor(80 GB / peak)` for
`--concurrent N` (expect N = 3–4 for the 0.6B student in bf16 with gradient
checkpointing).

## 1. Tier 1 — E1, the go/no-go (≈ 40 GPU-hours sequential)

```bash
# 1a. trihope first (its action shares feed random_routing), 1 seed to get the peak-memory number
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --only trihope
python -m analysis.run_report runs/baselines_small_v1        # writes action_share_by_phase.csv
# 1b. everything else, N at a time on the one GPU; --resume skips what is done
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --resume --concurrent 3
python -m analysis.run_report runs/baselines_small_v1
```

Read `runs/baselines_small_v1/analysis/{budget_curve.csv, pareto_budget.png,
forgetting_table.csv, action_share_by_phase.csv}`. Go/no-go (reframe §9):
`trihope` (or `trihope_no_hash`) on or above the Pareto front of
`molf_style`, `plateau_trigger`, `surprise_gate` at matched permanent
writes, and separated from `random_routing` with non-overlapping error
bars on worst retention delta.

## 2. Tier 2 — E5, bad teacher + rollback (≈ 30 GPU-hours)

```bash
python scripts/run_experiment.py configs/experiments/bad_teacher_small.yaml --concurrent 3
python scripts/rollback_teacher.py --run runs/bad_teacher_small_v1/trihope-seed1337 \
    --teacher math_teacher_corrupted --out runs/bad_teacher_small_v1/trihope-seed1337-rollback
python scripts/rollback_teacher.py --run runs/bad_teacher_small_v1/trihope-seed1337 \
    --teacher math_teacher_corrupted --baseline full_restore \
    --out runs/bad_teacher_small_v1/trihope-seed1337-fullrestore
# repeat for the other seeds and for gradient_routing
python -m analysis.run_report runs/bad_teacher_small_v1
```

Read `teacher_attribution.csv`, `containment.csv`, `containment_bars.png`,
and each `rollback_summary.json`.

## 3. Tier 3 — E2 / E3 / E4 (≈ 45 GPU-hours)

```bash
python scripts/run_experiment.py configs/experiments/r_tier_small.yaml --concurrent 3       # E2
python scripts/run_experiment.py configs/experiments/budget_sweep_small.yaml --concurrent 3 # E3 (1 seed / point)
python scripts/run_experiment.py configs/experiments/p_study_small.yaml --concurrent 3      # E4
python scripts/run_experiment.py configs/experiments/ablation_grid.yaml --concurrent 3      # signal ablations incl. stability_instant
python -m analysis.run_report runs/baselines_small_v1 runs/budget_sweep_small_v1 --out runs/figure1
python -m analysis.run_report runs/r_tier_small_v1
python -m analysis.run_report runs/p_study_small_v1
```

## 4. Optional — headline (1.7B) only if the GPU is idle after Tier 3

Three runs at 1 seed, sequentially (~6 h each):
`trihope`, `trihope_no_hash`, and whichever baseline was closest in E1.
Add them to `configs/experiments/headline.yaml` with the same overrides as
in `baselines_small.yaml`; `trihope_live_kd` is not part of the plan.

## Per-experiment write-up

After each `run_report`, write `runs/<experiment>/RESULTS.md`: the table,
the figure paths, and three sentences — what it shows, what it does not
show, anything surprising (task document, Part B, T8).

## Practicalities

- Each run is its own subprocess; one OOM kills one run. `--resume`
  continues interrupted runs bit-exactly from their latest checkpoint.
- `--concurrent N` shares the single visible GPU between N processes; use
  `--parallel-gpus N` instead when several GPUs are visible.
- `checkpoint.keep_tagged=0` (E5 manifests) keeps every `pre_merge` /
  `pre_phase_*` checkpoint; expect ~1.3 GB per tagged checkpoint for the
  0.6B student. Prune manually after the rollback runs.
- `random_routing` warns and falls back to uniform thirds if the shares
  file is missing — run `--only trihope` + `run_report` first.

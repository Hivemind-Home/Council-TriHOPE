# Rebuttal results, 2026-11-11 (rebuttal_v1 + rebuttal_volume)

This doc collects the results of the two rebuttal batches. Both ran on the A6000 box from commit `a8c73ca`. The volume batch also carries the patch in `results_live/rebuttal_volume_live/code.patch`.

- `rebuttal_v1`: 7 arms × seeds 1337/2024/7 = 21 runs. Manifest: `configs/experiments/rebuttal_v1.yaml`.
- `rebuttal_volume`: 4 volume-matched arms × 3 seeds = 12 runs. Manifest: `configs/experiments/rebuttal_volume.yaml`, whose header states the readings below before any number existed.

Per-run configs, metrics, commit traces, per-row eval losses and summaries are under `results_live/rebuttal_v1_live/` and `results_live/rebuttal_volume_live/`. Checkpoints and `events.jsonl` are not committed; they are on the box.

**M1** is the paper's metric: the mean over {general, code, math, medical} of the worst retention delta. Lower means less forgetting. Seeds are listed as 1337 / 2024 / 7. A difference is the arm minus its reference on the same seed.

## Bottom line

At an identical commit volume, *which* blocks the controller commits makes no measurable difference to forgetting:
- random selection forgets about the same as the controller;
- deliberately bad (lowest-agreement) selection forgets about the same in the shared-trainable regime, and less in the frozen regime.

The v1 gaps between the controller and the yoked random / anti arms were carried by commit volume, because those arms drifted to 1.4–4× the controller's commits. By the readings stated in `rebuttal_volume.yaml`, two things follow:
- the anti_commit positive-control claim is retracted;
- selection adds nothing measurable at gated volume.

What does hold: M1 tracks commit volume, and the controller commits few blocks. n = 3 seeds per arm, so no difference here is significant: the best possible sign test gives p = 0.125.

## Integrity checks (all pass)

| Check | Result |
|---|---|
| Config drift vs committed E1 trihope (`scripts/check_config_drift.py`) | 33/33 runs ok; only each arm's own overrides differ (`analysis/config_drift.txt`) |
| Shared parameters unchanged in every `shared_frozen_*` run (`scripts/weight_drift.py --expect-shared-unchanged`) | 18/18 runs: `changed_coords = 0` |
| v1 live-yoked arms commit exactly what their own controller would (`commit_schedule.py`) | ok, 12/12 |
| Volume arms commit the reference's per-step count (`commit_schedule.py --expect-match`) | total ratio 1.00 for all 12; worst single phase 0.984 |
| M1 recomputed from `eval_rows.jsonl` equals the logged retention deltas (`scripts/m1_rows.py`) | 0 mismatches over 33 runs |
| Reproduction: `trihope_e256` M1 on the paper's first 32 eval rows | 0.082 vs paper E1 trihope 0.0805 ± 0.0101 |

Note on the reproduction target: `REBUTTAL_RUNBOOK.md` and `REBUTTAL_CHANGES.md` called 0.084 "the paper's" trihope M1. That number is `random_commit` / the shared-only floor (`PREREG_SECONDARY_2026-09-25.md`). E1 trihope is 0.0805. The 0.082 result is within ±0.005 of both. The two docs are corrected in this PR.

## Manipulation check: the volume arms selected what they claim

Commit-weighted mean agreement (C̄) of the committed blocks (`scripts/commit_cbar.py`):

| Regime | Reference (controller) | schedule_random | schedule_anti |
|---|---|---|---|
| shared trainable | 0.644 / 0.599 / 0.606 | 0.588 / 0.478 / 0.495 | 0.465 / 0.414 / 0.416 |
| shared frozen | 0.634 / 0.636 / 0.645 | 0.547 / 0.505 / 0.547 | 0.425 / 0.434 / 0.467 |

## M1 per arm

Commits are totals per run.

| Arm | Regime | Commits | M1 @256 rows | mean | M1 @32 rows | mean |
|---|---|---|---|---|---|---|
| trihope_e256 (reference) | trainable | 56 / 89 / 110 | 0.085 / 0.070 / 0.076 | 0.077 | 0.087 / 0.077 / 0.082 | 0.082 |
| yoked_random (v1) | trainable | mean 119 | 0.105 / 0.086 / 0.115 | 0.102 | 0.108 / 0.089 / 0.122 | 0.106 |
| anti_commit (v1) | trainable | mean 246 | 0.127 / 0.150 / 0.147 | 0.141 | 0.127 / 0.147 / 0.152 | 0.142 |
| schedule_random | trainable | = reference | 0.091 / 0.075 / 0.079 | 0.082 | 0.096 / 0.077 / 0.083 | 0.086 |
| schedule_anti | trainable | = reference | 0.076 / 0.071 / 0.091 | 0.079 | 0.077 / 0.070 / 0.091 | 0.080 |
| shared_frozen_never_commit | frozen | 0 | 0.009 / −0.003 / −0.007 | −0.000 | 0.011 / −0.002 / −0.008 | 0.000 |
| shared_frozen_trihope (reference) | frozen | 477 / 553 / 498 | 0.052 / 0.068 / 0.093 | 0.071 | 0.055 / 0.075 / 0.094 | 0.075 |
| shared_frozen_yoked_random (v1) | frozen | mean 701 | 0.110 / 0.085 / 0.098 | 0.098 | 0.115 / 0.092 / 0.098 | 0.102 |
| shared_frozen_anti_commit (v1) | frozen | mean 2003 | 0.334 / 0.219 / 0.200 | 0.251 | 0.334 / 0.224 / 0.196 | 0.251 |
| shared_frozen_schedule_random | frozen | = reference (≥ 0.996) | 0.059 / 0.062 / 0.072 | 0.064 | 0.064 / 0.060 / 0.072 | 0.065 |
| shared_frozen_schedule_anti | frozen | = reference | 0.033 / 0.055 / 0.070 | 0.053 | 0.037 / 0.057 / 0.069 | 0.054 |

## Seed-paired differences (M1 @256, arm − reference)

| Comparison | Per seed | Same sign |
|---|---|---|
| schedule_anti − trihope_e256 | −0.009 / +0.000 / +0.015 | 2/3 |
| schedule_random − trihope_e256 | +0.006 / +0.005 / +0.003 | 3/3 (≈ 6 % of M1) |
| shared_frozen_schedule_anti − shared_frozen_trihope | −0.020 / −0.013 / −0.023 | 3/3, anti forgets **less** |
| shared_frozen_schedule_random − shared_frozen_trihope | +0.006 / −0.006 / −0.021 | 2/3 |
| anti_commit − trihope_e256 (v1, volume not matched) | +0.042 / +0.080 / +0.071 | 3/3 |
| yoked_random − trihope_e256 (v1, volume not matched) | +0.020 / +0.016 / +0.039 | 3/3 |
| shared_frozen_anti_commit − shared_frozen_trihope (v1) | +0.282 / +0.151 / +0.106 | 3/3 |
| shared_frozen_yoked_random − shared_frozen_trihope (v1) | +0.058 / +0.017 / +0.005 | 3/3 |
| shared_frozen_trihope − shared_frozen_never_commit | +0.043 / +0.071 / +0.100 | 3/3 |

## Against the readings stated before the runs (`rebuttal_volume.yaml`)

| Stated reading | Outcome |
|---|---|
| schedule_anti > reference in 3/3 seeds, both regimes → positive control holds | **Not met.** Trainable: no difference. Frozen: anti < reference, 3/3. |
| schedule_anti ≈ reference → v1 anti gap was volume; retract the positive-control claim | **Met.** Retract. |
| schedule_random > reference in 3/3 → controller selection beats random at identical volume | Trainable: 3/3, but +0.005 on M1 ≈ 0.077, n = 3. Frozen: not met. |
| schedule_random ≈ reference → v1 controller-vs-random gap was volume | **Met** in the frozen regime; the trainable effect is too small to claim. |
| volume ratio < 0.95 → report the shortfall | No shortfall (≥ 0.984 every phase). |

## Reproduce

```bash
python scripts/m1_rows.py results_live/rebuttal_v1_live results_live/rebuttal_volume_live
python scripts/commit_cbar.py results_live/rebuttal_volume_live/commit_schedule_*_seed*.json
python scripts/commit_schedule.py results_live/rebuttal_volume_live/schedule_*-seed7 \
    --reference results_live/rebuttal_v1_live/trihope_e256-seed7 --expect-match
```

The saved outputs of these commands are in `results_live/rebuttal_volume_live/analysis/`.

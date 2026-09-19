# E4 ablation grid — corrected, replay-ON, in-grid baseline

Generated 2026-09-19 from runs/ablation_grid_v1 (replay-ON, 13 arms, all `state=done, exit_code=0`, 6000 steps).

## Why this table replaces the earlier one

The previously reported ablation table compared **replay-OFF** ablation arms against a
**replay-ON** trihope number (0.188 vs ~0.119). That is not a like-for-like contrast, and it
produced the false claim that 11 of 12 ablations beat the shipped method. This grid carries its
own replay-ON `trihope` arm, so every delta below is measured against a baseline run under
identical settings. The corrected count is **4 of 12**.

## Method

`mean_forget` = mean over {general, code, math, medical} of `worst_retention_delta` (lower is better).

Baseline (in-grid, replay-ON `trihope`, seed 1337): **0.0961**

Noise floor: the only n=3 arm available (`consolidation_strict`, replay-OFF, seeds 1337/2024/7)
gives std of the mean-forgetting statistic ~0.0042, so a single-seed-vs-single-seed difference
carries ~**0.0059** of noise. Deltas are flagged significant only above 2x that.

| arm | replay-ON | delta vs baseline | x noise | verdict | replay-OFF |
|---|---:|---:|---:|---|---:|
| `no_repetition` | 0.0514 | -0.0448 | 7.6x | **beats baseline** | 0.0790 |
| `topm_2` | 0.0692 | -0.0270 | 4.6x | **beats baseline** | 0.0725 |
| `topk_25` | 0.0755 | -0.0207 | 3.5x | **beats baseline** | 0.0721 |
| `consolidation_strict` | 0.0909 | -0.0053 | 0.9x | within noise | 0.0701 |
| `trihope` | 0.0961 | +0.0000 | 0.0x | baseline | — |
| `no_volatility` | 0.0975 | +0.0014 | 0.2x | within noise | 0.0751 |
| `topk_100` | 0.1033 | +0.0071 | 1.2x | within noise | 0.0714 |
| `no_teacher_conf` | 0.1063 | +0.0102 | 1.7x | within noise | 0.0769 |
| `stability_instant` | 0.1195 | +0.0234 | 4.0x | hurts | 0.0985 |
| `topm_all` | 0.1392 | +0.0430 | 7.3x | hurts | 0.1173 |
| `no_surprise` | 0.1897 | +0.0936 | 15.9x | hurts | 0.0688 |
| `no_cosine` | 0.2078 | +0.1117 | 18.9x | hurts | 0.1498 |
| `moments_optimizer` | 0.2498 | +0.1536 | 26.0x | hurts | 0.0435 |

## What this says

**The two controller signals the method rests on are load-bearing.** Removing sustained cosine
(`no_cosine`, +0.112, 19x noise) or surprise (`no_surprise`, +0.094, 16x) degrades retention
sharply, and replacing the controller with raw Adam moments (`moments_optimizer`, +0.154, 26x)
is worst of all. That is the positive result the ablation is meant to establish, and it survives
the corrected baseline.

**Three ablations genuinely beat the shipped configuration**, and this should be reported, not
buried:

- `no_repetition` (-0.045, 7.6x noise) — the repetition signal is not merely inert, it is
  *counterproductive* for retention.
- `topm_2` (-0.027, 4.6x) and `topk_25` (-0.021, 3.5x) — both shrink the write budget, and both
  improve retention. The shipped config over-writes.

`consolidation_strict` (-0.005, 0.9x) is **within noise** — consistent with the separate n=3
finding that its earlier apparent effect was a tuning artifact, not a mechanism.

## Limits

- **n=1 per arm** (seed 1337). The noise floor is borrowed from a single other arm's seed spread
  in the replay-OFF grid; arms may differ in variance, and replay adds stochasticity of its own.
  The large deltas (>4x) are safe; the 2-4x band should be re-run at n=3 before it goes in a paper.
- `mean_forget` weights the four domains equally, which is a choice, not a given.
- The replay-OFF column is shown for context only; those runs have a different baseline and
  must not be read as a paired comparison.
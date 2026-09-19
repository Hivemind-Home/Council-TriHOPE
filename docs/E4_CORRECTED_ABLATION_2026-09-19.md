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
## Why the write budget matters: replay volume, not replay staleness

`replay_timing` (now computable — it needs `events.jsonl`, which the first mirror
pass excluded) separates the two candidate explanations for replay's retention
cost that §8 left open: a **late write landing on top** of newer learning, versus
a **stale cross-phase write-back**. Across the 13 grid arms:

| | correlation with mean forgetting |
|---|---:|
| replay **count** | **+0.74** pearson, +0.55 spearman |
| replay count, excluding the `moments_optimizer` outlier | +0.55 pearson, +0.43 spearman |
| **cross-phase share** | −0.17 pearson, −0.32 spearman |

Replay *volume* tracks forgetting; replay *staleness* does not — the cross-phase
share is low everywhere (1–12%) and if anything trends the wrong way. So the cost
is the quantity of write-back, not that the written-back rows are out of date.

The causal check is `no_repetition`, which does not appear in the replay table at
all because **it performed zero replays**. The repetition signal is what drives
R-store hits, and R-store hits are what trigger replay; disabling the signal
removes replay entirely, and that arm has the lowest forgetting in the grid
(0.0514 vs the baseline's 0.0961). This also explains why both write-budget
reductions win: `topm_2` (148 replays) and `topk_25` (194) sit below the
baseline's 255, and `moments_optimizer` — worst in the grid at 0.2498 — issues
1387, more than 5x the baseline.

This is the mechanism behind three of the four arms that beat the shipped
configuration, and it is consistent with §8's independent finding that replay is
the single dominant effect.

Caveat: 13 points, n=1 each, and a correlation across arms that differ in more
than replay count. `no_repetition` is the closest thing to a controlled test and
it points the same way, but a deliberate replay-rate sweep at n=3 would settle it.

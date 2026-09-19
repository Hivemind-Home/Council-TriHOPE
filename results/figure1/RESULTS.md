# E3 / E4 / ablations

## E3 — threshold sweep (`runs/budget_sweep_small_v1`, 22 runs, n=1)
Pooled with E1 into `runs/figure1/pareto_budget.png` — the paper's identity
figure.

Best points all use `repetition_low=0.2`: `trihope_s4_r0p2` 0.1302 @ 1.70e8,
`trihope_s1_r0p2` 0.1320 @ 1.92e8, `trihope_s4_r0p3` 0.1344 @ 2.04e8, against
shipped-threshold `trihope_s1_r0p3` at 0.1888. Consistent with the STOP-1
finding that repetition, not surprise, does the discriminating.

**The sweep also contains the result that refutes E1's headline:**
`molf_style_a0p5` (`adam_score_rule=snr_threshold`, top-k inherited at 0.5)
reaches 0.0796 @ 1.01e8 at n=1, and 0.0841 +/- 0.0062 @ 1.03e8 when re-run at
3 seeds — better than `trihope` on retention *and* write count.

## E4 — when permanence fires (`runs/p_study_small_v1`, 5 runs, n=1)

| spec | worst ret D | perm writes |
|---|---|---|
| `p_off_control` | 0.0913 | 0 |
| `p_study` | 0.0943 | 3.37e8 |
| `p_forced_bad` (step 2075, mid `novel_inject`) | 0.1202 | 6.67e8 |
| `p_forced_plausible` (2350) | 0.1207 | 6.83e8 |
| `p_forced_lowconf` (3850, medical) | 0.1289 | 6.45e8 |

Timing of permanence matters — forcing merges at the wrong moment costs ~28%
more forgetting, worst when confidence is low, which supports the gate. But
`p_off_control` ~ `p_study` means P as configured barely helps retention.

## Ablations (`runs/ablation_grid_v1`, 12 runs, n=1)

| ablation | worst ret D | perm writes | vs `trihope` 0.1883 |
|---|---|---|---|
| `consolidation_strict` | 0.1153 | 8.81e7 | **-0.073, dominates both axes** |
| `no_surprise` | 0.1169 | 1.79e8 | -0.071 |
| `no_repetition` | 0.1339 | 4.50e8 | -0.054 |
| `no_teacher_conf` | 0.1412 | 1.51e8 | -0.047 |
| `stability_instant` | 0.1852 | 6.23e8 | -0.003 |
| `no_cosine` | **0.3544** | **1.79e10** | **+0.166** |

**What it shows.** `C_bar` (sustained cosine) is the load-bearing mechanism:
removing it is the only ablation that hurts, and it does so catastrophically
— 200x the permanent writes, because nothing gates consolidation any more.
`consolidation_strict` (`min_stability_C=0.7`, `min_repetition=0.6`,
`period=1000`) beats the tuned configuration on retention *and* write count,
so the mechanism is sound and the tuning is not.

**What it does not show.** That the other signals help. 11 of 12 ablations
improve on `trihope`. `moments_optimizer` (0.0906) and `topm_all` are
dead-signal controls with 0 permanent writes — trivially unable to forget
from permanent writes, not real wins.

**Caveat.** E3/E4/ablations are n=1 by manifest design, unlike E1/E2/E5's 3
seeds. Strong indications, not established effects.

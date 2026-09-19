# E2 — does the R tier earn its place (`runs/r_tier_small_v1`)

15 runs, 5 specs x 3 seeds, 0 failures.
Tables: `analysis/budget_curve.csv` (incl. `steps_to_recover_code`),
`action_share_by_phase.csv`

| spec | worst ret D | +/- | permanent writes |
|---|---|---|---|
| `trihope_r_terminal` (R parks, never replayed) | 0.1194 | 0.0100 | 1.99e8 |
| `fp_only` (no R tier at all) | 0.1211 | 0.0030 | 1.95e8 |
| `trihope_replay` (full trihope) | 0.1939 | 0.0111 | 1.36e8 |
| `trihope_no_hash_replay` | 0.2321 | 0.0486 | 4.22e8 |
| `p_only` (everything permanent) | 0.9144 | 0.3588 | 2.57e11 |

**What it shows.** Deferral is free; replay is not. `r_terminal` (parks rows
and never reads them back) and `fp_only` (no R tier) are statistically
indistinguishable at ~0.12, while `trihope_replay` — identical except that
parked rows are replayed into F — sits at 0.194. Replaying deferred examples
reintroduces the interference deferral was meant to avoid, costing 62% more
forgetting. `p_only` (0.914, 2.57e11 writes) is the upper control: routing
everything to permanent storage is catastrophic, so the F tier is essential.

**What it does not show.** That replay is worthless. It buys a 32% reduction
in permanent writes (1.36e8 vs ~1.97e8), so neither point dominates on a raw
Pareto plot. But on the paper's own metric — forgetting per permanent write —
terminal deferral wins, 0.60e-9 vs 1.43e-9.

**Surprising.** The result replicates four independent ways: `r_terminal`
0.119, `fp_only` 0.121, E1's `no_retrieval` 0.119, all against ~0.19 for
every replay-enabled configuration.

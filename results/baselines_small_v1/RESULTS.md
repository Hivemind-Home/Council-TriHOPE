# E1 — controller comparison (`runs/baselines_small_v1`)

33 runs, 11 specs x 3 seeds (1337/2024/7), 0 failures. Cache mode, 6000-step
stream, thresholds `surprise_high=0.75`, `repetition_low=0.3`,
`stability_high_C=0.30` (STOP-1 retune, see `docs/STATUS.md` §8).

Figures: `analysis/pareto_budget.png`, `analysis/retention_curves.png`
Tables: `analysis/budget_curve.csv`, `forgetting_table.csv`,
`action_share_by_phase.csv`, `teacher_attribution.csv`

| spec | worst ret D | +/- | permanent writes |
|---|---|---|---|
| `gold_ce` | 0.1144 | 0.0020 | 2.04e8 |
| `no_retrieval` | 0.1194 | 0.0091 | 2.11e8 |
| **`trihope`** | **0.1883** | **0.0036** | **1.31e8** |
| `no_consolidation` | 0.1921 | 0.0056 | 0 |
| `trihope_no_hash` | 0.2322 | 0.0502 | 3.95e8 |
| `surprise_gate` | 0.3506 | 0.1459 | 0 |
| `plateau_trigger` | 0.3591 | 0.1090 | 0 |
| `lora_only` | 0.3680 | 0.1141 | 0 |
| `random_routing` | 0.6500 | 0.1025 | 2.13e8 |
| `molf_style` | 0.8115 | 0.1386 | 2.23e12 |
| `full_ft` | 5.1060 | 2.7793 | 2.64e12 |

**What it shows.** The routing signals carry information: `random_routing`
replays trihope's own per-phase action shares with the assignment shuffled —
same budget, comparable write count (2.13e8 vs 1.31e8) — and forgets 3.5x
more, with error bars nowhere near touching. `trihope_no_hash` is worse than
`trihope`, so the bucket-id counter earns its place.

**What it does not show.** That `trihope` beats a fairly-configured baseline.
`molf_style` here runs `adam_score_rule=epd_argmax` with
`top_k_fraction=1.0` — every coordinate open — while `trihope` runs at 0.5.
The 4.3x margin is a budget artefact. The fair variant already in
`budget_sweep_small.yaml` beats `trihope` 0.0841 +/- 0.0062 vs 0.1883 +/-
0.0036 (see `docs/STATUS.md` §2a). `surprise_gate` separation is
convention-dependent: non-overlapping under population sd, overlapping under
the sample sd `run_report` uses.

**Surprising.** Four specs beat `trihope` on retention, and three of them are
ablations of it (`no_retrieval`, and in E2 `r_terminal`/`fp_only`). Removing
machinery improves the metric the method exists to optimise.

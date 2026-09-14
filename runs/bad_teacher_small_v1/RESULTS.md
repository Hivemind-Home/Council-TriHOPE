# E5 — containment and rollback (`runs/bad_teacher_small_v1`)

21 runs, 7 specs x 3 seeds, plus rollback replays. 0 run failures.
Half the math buckets in `math_recurrent` carry shuffled answers under the
tag `math_teacher_corrupted`.
Tables: `analysis/containment.csv`, `budget_curve.csv`,
`teacher_attribution.csv`

Where the corrupted teacher's updates went:

| spec | P share | coords direct to base | perm writes (all teachers) |
|---|---|---|---|
| `trihope` | **0.00%** | **0** | 1.35e8 |
| `trihope_lowconf` | 0.00% | 0 | — |
| `gradient_routing` | 0.00% | 0 | — |
| `molf_snr` (fair MoLF, added 2026-09-14) | **0.00%** | **0** | **8.18e7** |
| `molf_style` (epd, `top_k_fraction=1.0`) | 80.98% | 2.73e11 | 2.18e12 |

**What it shows.** Containment is real. `trihope` routed none of the corrupted
teacher to permanent storage and wrote zero coordinates directly into base
weights. `rollback_teacher.py` reports *"no consolidation event attributes
>= 0.50 of a merge to 'math_teacher_corrupted'; nothing to roll back
(containment held)"* — a result, not an error (guide §5). The confidence gate
tightens attributed merges 300x (`trihope` 4.39e6 -> `trihope_lowconf`
1.46e4).

**What it does not show.** That containment distinguishes `trihope`. The fair
MoLF baseline contains identically — 0% to P, 0 coords direct — while writing
40% fewer permanent coordinates overall. It is not trivially abstaining: it
writes 8.18e7 coords permanently, it simply never routes the corrupted
teacher there. The 81% figure belongs solely to the `top_k_fraction=1.0`
variant, which writes *everything* to P.

**Surprising.** `full_ft` and `lora_only` show 0 corrupted-teacher actions
because their controller is off and no decisions are logged — those rows mean
"not applicable", not "perfect containment", and should not be read as such.
The `trihope` rollback replays in the original pass were truncated at ~3750
by the resume bug (`f27732e`); they are regenerated in
`runs/_prefix_resume_bug_*` vs the current dirs.

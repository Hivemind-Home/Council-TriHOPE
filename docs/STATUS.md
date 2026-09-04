# STATUS — ICLR-2027 push

Branch `iclr2027-push` (from `main` @ `8dc4917`). Every task is a commit;
details per task in `docs/CHANGELOG_iclr.md`; the campaign commands in
`docs/GPU_RUNBOOK.md`. Result slots below are filled in from `runs/` once
the GPU campaign has run — nothing has been run on the real stream yet.

## 1. Task table

| Task | Status | Commit | Tests after |
|---|---|---|---|
| T0 environment + baseline sanity | done (CPU: pytest, ruff, `stream_smoke` + resume on real data) | `6f2580d` (in T1) | 420 |
| T1 sustained C̄ in the policy, P fires | done | `6f2580d` | 423 |
| T2 R→F replay promotion | done (micro-step deviation) | `747b0c8` | 447 |
| T3 four E1 baseline controllers | done | `747b0c8` | 447 |
| T4 corrupted-teacher stream | done (batch-coherent deviation) | `a95270c` | 473 |
| T5 attribution + rollback | done (adapter-reset deviation) | `149fd4b` | 488 |
| T6 gradient-routing baseline | done | `29d19fd` | 494 |
| T7 budget curve + Pareto | done | `b95237a` | 454 |
| T8 manifests + runbook + `--concurrent` | done | `2e83cff` | 458 |
| T9 Part C reading notes + reading-driven fixes (EPD rule, peak precondition, pre-merge bug) | done | see git log | 498 |
| T10 this file | skeleton; results pending | — | — |

Real-data checks done on this CPU box (`stream_smoke`, Qwen tokenizer,
math + medical Layer-C): replay fires (22 rows, none in `novel_inject`),
resume from step 50 is bit-exact with replay on, the corrupted stream
attributes decisions to the tag and changes the digest, gradient routing
+ slice ablation run, and the selective-rollback / full-restore scripts
execute end to end on a smoke run (see the changelog for numbers).

## 2. E1 — controller comparison (`runs/baselines_small_v1`)

**Pass condition (reframe §5 E1, §9):** `trihope` (or `trihope_no_hash`)
on or above the Pareto front of `molf_style`, `plateau_trigger`,
`surprise_gate` at matched permanent writes, and clearly separated from
`random_routing` (non-overlapping error bars on worst retention delta).

- Table: `runs/baselines_small_v1/analysis/budget_curve.csv` — _pending_
- Figure 1: `runs/baselines_small_v1/analysis/pareto_budget.png` — _pending_
- Action shares: `action_share_by_phase.csv` — _pending_
- Verdict: _pending_ (go / conditional go / no-go)

## 3. E5 — containment and rollback (`runs/bad_teacher_small_v1`)

**Pass condition (reframe §5 E5):** most of the corrupted teacher's
influence stays out of P; TriHOPE's damage on general/code/medical is
lower than full FT and MoLF-style at matched budget; selective rollback
recovers more than a full restore.

- Containment: `analysis/containment.csv`, `containment_bars.png` — _pending_
- Damage: `analysis/budget_curve.csv` (retention deltas per spec) — _pending_
- Rollback: `*-rollback/rollback_summary.json` vs `*-fullrestore/rollback_summary.json` — _pending_
- Verdict: _pending_

## 4. E2 / E3 / E4

- E2 `runs/r_tier_small_v1/analysis/budget_curve.csv` (steps-to-recover on `code_revisit`) — _pending_
- E3 `runs/figure1/pareto_budget.png` (E1 + sweep pooled) — _pending_
- E4 `runs/p_study_small_v1/analysis/{p_selection_stats,damage_recovery}.csv` — _pending_

## 5. Unable to do / not done

- **`stable_gate` baseline (STABLE, arXiv:2510.16089) — not implemented.**
  The closest prior statement of "permanence as a budgeted decision" gates
  each LoRA→base merge with an outcome-level probe (EM / bits / KL on
  previously edited anchors, binary search over the adapter scale α).
  Faithful spec: anchors = rows already consolidated into P; at each
  candidate merge compute the per-token KL of adapted vs base on the
  anchors against a budget ε; binary search α ∈ [0.1, 1] with ≤ 5
  evaluations; merge α·ΔW or reject; no use of g/m/v; report the extra
  forward-pass cost. ~1 day of work; reviewers are likely to ask for it.
- **MoLF's universal momentum tracking** (losers' moments advancing on a
  shared clock) is deliberately not reproduced in `molf_style` — it
  contradicts Theorem 1 / MaskedAdamW.

- No run on the real stream: this machine is CPU-only. Everything above
  is validated on `stream_smoke` (tiny in-repo student, real data).
- `stream_headline` (1.7B) is out of the plan on one GPU; the runbook
  lists it as an optional last tier.
- `trihope_live_kd` (live 1.5B teachers) dropped: cache mode only.
- Part C reading notes depend on the papers being reachable at their
  arXiv ids; any paper that could not be fetched is marked in
  `docs/related_work_notes.md` rather than summarised from memory.

## 6. Deviations from the task document (all recorded in the changelog)

1. **T2** replay is an extra micro-step inside the triggering step, not a
   substitution of the next stream batch (keeps `step == stream index`);
   recurrence counts distinct steps; replayed HF rows are CE-only.
2. **T4** corruption selects whole bucket subsets + whole background
   steps (batch-coherent), applied by row through a dataset view, with
   the cross-phase leak reported; `confidence` defaults to the row's own.
3. **T5** rollback also resets the blocked teacher's dominated adapters;
   `full_restore` skips the phase via `train.skip_step_ranges` (digest
   unchanged; skipped steps skip eval); merges without attribution are
   ignored; ledger completeness is asserted as ⊆ plus the merged-block
   exception.
4. **T6** the slice ablation zeroes instead of re-initialising.
5. **T7** `permanent_writes` are (coordinate, step) write events on base
   weights so full FT, LoRA-only and merge-based controllers share one axis.
6. **T8** `p_only` = `always_p`; `surprise_gate` / `random_routing` run
   with replay on; `trihope_no_hash` added as a headline candidate;
   `--concurrent N` added for the single-GPU campaign.
7. **T2+T3** share one commit (same hunks); T0's numbers live in T1's commit.

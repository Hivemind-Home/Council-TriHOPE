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
| T10 this file | done (results pending the GPU campaign) | see git log | 498 |

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

### 4b. Tier 4 — live logit-KD (`runs/live_kd_small_v1`, appendix)

**Pass condition:** `trihope_live` shows the same qualitative routing as
cache-mode `trihope` (R in `novel_inject`, F rising through the recurrent
phases, P at the 250-step sweeps, zero replays in `novel_inject`) and the
E1 ranking of trihope vs the trigger baselines on worst retention delta is
unchanged in sign.

- Table: `runs/live_kd_small_v1/analysis/{action_share_by_phase,forgetting_table}.csv` — _pending_
- Verdict: _pending_

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
- Live logit-KD is not in the figures: the campaign is sequence-level
  distillation on cached text (`loss/kd` = 0). The Tier 4 manifest
  `live_kd_small.yaml` (added 2026-09-05) runs the headline controller and
  its E1 rivals with four live Qwen-vocabulary teachers as an appendix
  robustness check — slot in §4b.
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

## 7. Paper wording fixes from the 2026-09-05 data / objective audit

Recorded in full in `docs/data_provenance.md`; the paper must:

1. Call the main-result objective **sequence-level distillation from
   cached teacher traces**; reserve "temperature-scaled KL" for the Tier 4
   appendix run where it is actually active.
2. Describe `teacher_confidence` as a dataset-provided per-answer score
   whose derivation differs by domain (logit-derived for math/medical,
   undocumented scoring pass for general/code).
3. Give, per domain, the source corpus and the authoring model as far as
   documented: general and code traces were shipped with public corpora
   (R1-Distill-Llama-70B via Glaive; DeepSeek-R1 via NVIDIA); math and
   medical were generated for this project (models per `teacher_id`,
   generation logs still to be obtained). A hosted model
   (`deepseek-v4-flash`) graded curation samples only.
4. State the truncation: math traces cut at ~1000 characters; student
   context 512 tokens (1024 headline), so the student sees the opening of
   a trace.
5. Keep teacher selection out of the contributions (cache mode routes by
   the row's `teacher_id`; the cosine router runs only on synthetic data).
6. Say where the evidence comes from: the controller tracks its own
   bias-corrected Adam-style moments for every indexed coordinate (a 1/8
   sketch), identical to Adam's state on always-open coordinates and alive
   on closed ones; the optimizer's state stays exactly masked. Credit
   MoLF's universal momentum tracking for the idea; "zero optimizer
   overhead" becomes "no change to the update rule".

## 8. 2026-09-05 — the campaign restarts from E1

`scripts/diagnose_p.py` on the first cache-mode `trihope` seed (2 000 steps,
9 384 decisions): C̄ = 0.000 at every decision, surprise pinned at 20.0,
P never fired, zero consolidations. Cause (`controller/moments.py`
docstring): under exact masking a never-opened base-weight module has
m = v = 0 forever, base modules dominate Top-M by gradient norm, and P —
the only action that opens them — needs those signals. Every E1 run made
before commit `5788dac` routed on repetition alone and must be
deleted. Fix: `controller.moments.source=tracked` (default; the old
behaviour is the `moments_optimizer` ablation). The live-KD tier had a
second, independent bug (KD mask zeroed; fixed in `f9b422d`). Both VMs:
`git pull`, delete `runs/baselines_small_v1` and `runs/live_kd_small_v1`,
relaunch from step 4 of `docs/RUN_PIPELINE_2026-09-04.md`.

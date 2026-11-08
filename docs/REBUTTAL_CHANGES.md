# ICLR-2027 rebuttal: what changed in the code, and the commands to run (2026-11-08)

> For the co-author who will study the branch and hand it to the GPU operator.
> Companion pages: **docs/REBUTTAL_RUNBOOK.md** (the operator's page: step list and
> outcome table) and this page (what was changed, why, where, how to check it).

Branch **`rebuttal/iclr2027-controls`** on `origin`, two commits on top of `main`
`1ed99a3` (the commit the submitted paper was built from):

| commit | content |
|---|---|
| `bb503c6` | the controls, the trainability mode, the traces, manifests, scripts, tests, runbook |
| `9bc0d83` | `run_rebuttal.sh`: `python3` fallback, smoke-check guard |

Nothing on the branch changes an existing arm. The new code is reached only through
three new config values (`controller.debug.policy_override=yoked_random|anti_commit`,
`train.trainable=routed`, `eval.row_losses=true`) and two new trace files that are
written next to `events.jsonl`. `train.trainable=all|lora`, every existing override,
and every number in the paper are unaffected. Test suite: 551 tests (was 514), all
green on CPU.

---

## 1. Why: the three reviews in one sentence each

All three reviews (ratings 2/2/2) accept the mechanism and the audit framing and
reject the **inferential weight of the null result** ("selection adds nothing
measurable"):

| Reviewer | The objection, in their terms | What would answer it |
|---|---|---|
| BDmh (W1, W2/Q2) | The shared parameters (tied embedding/LM head, norms: 26 % of the model) train every step outside the gate and drift 65x further than the routed weights. Reference = random = shared-only at M1 0.084, so the routed commits may add nothing *on top of a floor*. And there is no positive control: nothing shows the benchmark can detect a selection effect at all. | A regime with the shared parameters frozen; a deliberately bad selector at the controller's volume. |
| 1Ume (W1/Q1, W3/Q3) | `random_commit` matched commit *rate per phase*, not per-step count or timing (89 % of controller commits fall in the first 200 steps of a phase). Retention measured on 32 rows, no per-row uncertainty. MoFO's Random-BCD contradicts the related-work claim. | A per-step timing-matched random control; a larger eval with per-row losses; a text concession. |
| ppjp | Low-confidence copy of the above. | Covered by the same arms. |

The branch answers the experimental part. The text part (MoFO concession, softening
"exact isolation", presentation fixes) is paper-side and not on this branch.

---

## 2. The three mechanisms

### 2.1 Live-yoked controls: `policy_override = yoked_random | anti_commit`

**Idea.** On every step the policy runs the *full* controller decision on the Top-M
selected blocks, obtaining the controller's labels. It then keeps every `R`
(deferral, so the replay it triggers is the controller's own), counts the controller's
`P` labels (`k_t`), and re-seats exactly `k_t` commits on a different subset of the
non-deferred blocks:

- `yoked_random`: `k_t` blocks drawn uniformly without replacement from the
  non-deferred selected blocks. The controller's own picks are *eligible*, so the
  overlap with the controller is a measured quantity (smoke: 0.57), not a design choice.
- `anti_commit`: the `k_t` non-deferred blocks with the **lowest** agreement, read from
  the same source the P rule uses (`PolicyConfig.stability_source`: C̄ sustained, or C),
  ties broken by Top-M rank. The controller commits the *highest*-agreement blocks, so
  this is the mirror-image policy at identical volume and timing. This is the positive
  control: if a deliberately bad selector at the controller's volume does not forget
  more, the benchmark cannot resolve selection at that volume.

Every other non-deferred block gets `F`. Consequences that matter for the rebuttal:

- **Count and timing match the controller step for step** within the arm's own
  trajectory. No reference run is read; the arm is self-contained and all 21 runs can
  run concurrently. (The paper's `random_commit` needed the reference's per-phase
  shares; `quota_shuffle` drifted to ~2x volume. Neither applies here.)
- The matching is *within-trajectory*: the yoked arm's weights diverge from the
  reference arm's after the first re-seated commit, so its `k_t` sequence is its own
  controller's, not the reference run's. This is the right comparison for 1Ume's
  question ("same per-step count, random identity") and `scripts/commit_schedule.py`
  reports the volume ratio against `trihope_e256` so the reviewer can see how close
  the two trajectories stayed.
- Deterministic: the random draw uses a fresh `torch.Generator` seeded by
  `(train.seed, step, sha1(phase))`. It never touches the global RNG, so the rest of
  training is bit-identical to what the reference arm does with the same seed up to the
  first divergent commit; it is identical on every DDP rank; it is bit-exact under
  checkpoint resume (no carried state). `anti_commit` draws nothing.
- Gates are preserved: the confidence gate, `force_r`, `disable_stores`, and the
  blocked-teacher demotion all act on the controller labels *before* re-seating, so a
  step on which the controller would not commit never commits in the control either.

**Where.** `src/hivemind/controller/policy.py`:
`_phase_salt()` (line 28), `StoreAction.controller_store` (line 106, the controller's
label is kept on every action for the trace), the dispatch in `decide()` (lines
205-225), `_controller_label()` (line 288: the `override is None` branch of `decide()`
factored out so the two cannot drift apart), `_yoked_labels()` (line 309).
`src/hivemind/controller/config.py`: `DebugConfig` accepts the two new names (a typo
fails at launch).

**Verified by.** `tests/test_rebuttal_controls.py`: same `k`, same `R` set, picks
actually move, deterministic in `(seed, step, phase)`, identical labels when `k=0`;
`anti_commit` picks the lowest-C̄ blocks and follows `stability_source`; the four gates.
CPU smoke (70 steps, seed 1337): `trihope` 15 commits, `yoked_random` 21 (overlap with
controller 0.57), `anti_commit` 24 (overlap 0.21); on both yoked arms
`controller_commits == commits` (the invariant `commit_schedule.py` enforces).

### 2.2 Shared-frozen regime: `train.trainable = routed`

**Idea.** Freeze the `shared` parameter group only (tied embedding/LM head and every
norm, including Qwen3's q_norm/k_norm) and train only what the gate governs: the P
base weights of the routed blocks and the F LoRA adapters. With the floor removed, any
difference between controller, yoked-random and never-commit is attributable to the
routed writes. `shared_frozen_never_commit` (`disable_stores=[P]` in this regime) is
BDmh's "ideally also never-commit": the floor of the frozen regime.

**Where.** `src/hivemind/training.py` lines 559-581: `requires_grad=False` on
`student.get_param_groups()["shared"]`, applied **before** DDP wrap and **before**
`build_optimizer` (the optimizer factory drops groups with no trainable parameter, so
no stale group). Both backbones return the groups `{"P","F","shared"}`; `shared` is
built from `self.parameters()`, which de-duplicates by identity, so a tied
embedding/LM head is one `Parameter` and the freeze covers the head. The HF backbone
never enables gradient checkpointing (the config key exists but nothing consumes it),
so the "frozen embeddings + re-entrant checkpoint = no gradients" trap cannot occur.

**Proof in the output.** A SHA-256 digest of every shared tensor is taken at
construction (`_param_digests`, line 233) and again after the last optimizer step
(`_shared_frozen_check`, line 254, main rank, raw module). `run_summary.json ->
extra.shared_frozen_check = {tensors, coords, changed_tensors, unchanged}`. On the
0.6B student this is 6 tensors on the smoke model, more on Qwen3 (one per norm plus the
embedding). `scripts/weight_drift.py --expect-shared-unchanged` is the independent
second check: it compares the checkpoint with the pretrained weights and exits non-zero
if any shared coordinate moved.

**Caveat the operator is told about.** On the CPU smoke student (randomly initialised,
tiny) freezing the embeddings leaves nothing to learn with: the frozen arms sit at loss
~11 and commit 0. That is expected there and says nothing about the pretrained 0.6B.
The runbook's `early` stage exists so the operator confirms, about an hour in, that
`shared_frozen_trihope` on the real model shows a falling loss and commits > 0 by the
end of `code_recurrent` (step 1999). If it does not, the batch is stopped and reported.

### 2.3 Traces: `commits.jsonl` and `eval_rows.jsonl`

Both are `EventTrace` files next to `events.jsonl`, written on the main rank only,
flushed every record (so a resumed run loses nothing), append-mode (a resumed run may
repeat steps; every reader keeps the last record per step).

`commits.jsonl`: one record per trained step. Non-committing step:

```json
{"step": 1, "phase": "math_warm", "r": 4, "f": 0, "p": 0}
```

Committing step (and, on yoked arms, what the controller would have committed):

```json
{"step": 22, "phase": "math_recurrent", "r": 0, "f": 3, "p": 1,
 "modules_p": ["L0.ffn.P"], "C_bar_p": [0.6121], "C_p": [0.7814],
 "controller_p": ["L1.attn.P"]}
```

This is what gives the reviewer the commit-time distribution (1Ume), the per-phase
volume ratio between arms, the overlap between a yoked arm and the controller, and the
C̄ of every committed block. `scripts/commit_schedule.py` reduces it; written at
`training.py` lines 1559-1578.

`eval_rows.jsonl` (`eval.row_losses=true`): one record per eval pass and domain with
the per-row token-summed loss, token count and bucket, in loader order:

```json
{"step": 59, "phase": "medical_recurrent", "domain": "math",
 "loss_sum": [819.38, 869.04, ...], "tokens": [128, 149, ...],
 "bucket": ["math_calculus_hard", "math_calculus_hard", ...]}
```

`loss_sum / tokens` per row reproduces the aggregate per-token loss exactly, so a
per-row bootstrap of M1 is a few lines of numpy. The eval runs 256 rows per domain in
the rebuttal manifest (`eval.max_batches=128` x batch 2); rows come in loader order, so
**the paper's 32 rows are the first 32 of each record** and the 32-row M1 can be
recomputed for continuity. Code: `src/hivemind/evaluation.py` (`EvaluationConfig.row_losses`,
`_eval_one_domain(keep_rows=)`, `run_evaluation(row_sink=)`); the sink is wired at
`training.py` line 1978.

---

## 3. The experiment: `configs/experiments/rebuttal_v1.yaml`

7 arms x 3 seeds (1337, 2024, 7) = 21 live runs on `stream_small`, launched in this
order (the runner is in manifest order, so if time runs out the last arms drop):

| # | arm | `train.trainable` | override | role |
|---|---|---|---|---|
| 1 | `anti_commit` | all | `anti_commit` | positive control in the paper's regime |
| 2 | `shared_frozen_trihope` | routed | - | controller, floor removed |
| 3 | `shared_frozen_yoked_random` | routed | `yoked_random` | the comparison both reviewers asked for |
| 4 | `trihope_e256` | all | - | reference under the 256-row eval; its first-32-row M1 must reproduce 0.084 +- 0.005 |
| 5 | `yoked_random` | all | `yoked_random` | 1Ume's timing-matched control, paper's regime |
| 6 | `shared_frozen_anti_commit` | routed | `anti_commit` | positive control, frozen regime |
| 7 | `shared_frozen_never_commit` | routed | `disable_stores=[P]` | floor of the frozen regime |

Manifest conventions worth knowing when reading it:

- Every arm carries `_pin` (`controller.policy.surprise_high=1.25`,
  `controller.policy.repetition_low=0.40`, `controller.retrieval.replay_on_hit=true`):
  the paper's thresholds, pinned explicitly so a box-local edit of `stream_small.yaml`
  cannot change them. And `_eval` (`eval.max_batches=128`, `eval.row_losses=true`).
- `run_experiment.py` prefixes every override with `++` (add-or-override), which is why
  `eval.row_losses`, absent from the base YAML, is accepted under Hydra struct mode.
- `--live` appends `_live` to the experiment name: runs land in
  `runs/rebuttal_v1_live/<arm>-seed<seed>/`, and `run_rebuttal.sh` derives the same
  path from the manifest.
- `configs/experiments/rebuttal_smoke.yaml` is the same 7 arms on `stream_smoke`
  (70 steps, CPU) and is what the `smoke` stage runs.

Cost: each run ~1-2 h with three sharing an A6000/A100-80G; the 256-row eval adds
~10-15 min per run. Whole batch ~12-16 h at `CONCURRENT=3`.

---

## 4. Commands to give the operator

Verbatim, in this order. The one-page version with the pass/fail criteria is
`docs/REBUTTAL_RUNBOOK.md`; nothing below needs a decision from the operator.

```bash
cd <the checkout that ran the live campaign>      # e.g. /home/a6000/asif/Council-TriHOPE
git fetch origin
git checkout -B rebuttal/iclr2027-controls origin/rebuttal/iclr2027-controls
pip install -e ".[dev,data]"                       # only if the env is new
```

```bash
bash scripts/run_rebuttal.sh check
```
~10 min, no GPU work. Prints branch/commit (must be `rebuttal/iclr2027-controls` at
`9bc0d83` or later), warns on uncommitted changes under `src/ scripts/ configs/ tests/`,
runs the tests that cover the new paths (all green), preflight, and the 21-row dry-run
whose override columns should match the table in section 3.

```bash
bash scripts/run_rebuttal.sh smoke
```
~20-30 min on CPU; the GPU is free meanwhile. Seven 70-step runs. Pass criteria: no
line ends in `<-- FAIL`; every `shared_frozen_*` line says `shared_unchanged=True`; the
commit-schedule table shows `ctl` = `commits` on `yoked_random` and `anti_commit`.
The frozen arms committing 0 here is expected (section 2.2).

```bash
bash scripts/run_rebuttal.sh launch
```
The 21 runs, three at a time, `--live --resume`. Safe to interrupt and re-run: finished
runs are skipped, interrupted ones resume from their last checkpoint. `CONCURRENT=2`
on a 40 GB card. Progress in `runs/rebuttal_v1_live/*/status.json` and each run's
`stdout.log`. A `provenance_<ts>.txt` (commit, status, torch version) is written in
the experiment dir at launch.

```bash
bash scripts/run_rebuttal.sh early          # second shell, ~1 h after launch
```
One line per run: step, loss at start -> now, committing steps so far. Stop criterion:
a `shared_frozen_*` run whose loss is flat, or `shared_frozen_trihope` with 0 commits
after step 1999. Report; do not restart.

```bash
bash scripts/run_rebuttal.sh post
```
Also fine on a partial batch. Runs the config-drift check against the committed live
E1 results, `weight_drift.py --expect-shared-unchanged` on every frozen arm (non-zero
exit means a shared coordinate moved: **report, do not re-run**), plain drift on the
others, `commit_schedule.py` per seed against `trihope_e256`
(`runs/rebuttal_v1_live/analysis/commit_schedule_seed*.json`), `analysis.run_report`,
and a tarball `rebuttal_results_<ts>.tgz` of the run dirs without checkpoints.

What to send back: the tarball (or `runs/rebuttal_v1_live` minus `checkpoints/` on a
results branch) plus the console output of `check`, `smoke` and `post`.

`bash scripts/run_rebuttal.sh all` runs check + smoke + launch + post unattended.
Environment knobs: `CONCURRENT`, `PY` (defaults to `python`, then `python3`),
`MANIFEST`.

---

## 5. Commands for us, once the results are back

```bash
# commit timing / volume / overlap, one seed, with the pass gate on volume
python scripts/commit_schedule.py runs/rebuttal_v1_live/*-seed1337 \
    --reference runs/rebuttal_v1_live/trihope_e256-seed1337 --expect-match
```

```bash
# the frozen-shared proof, independent of run_summary
python scripts/weight_drift.py runs/rebuttal_v1_live/shared_frozen_*-seed* --expect-shared-unchanged
```

```bash
# aggregate tables / figures
python -m analysis.run_report runs/rebuttal_v1_live
```

Per-row bootstrap of M1 from `eval_rows.jsonl`: take the last record per
`(step, domain)` for the retention eval steps, `loss = loss_sum / tokens` per row,
first 32 rows = the paper's eval, all 256 = the new one, resample rows within domain,
recompute the worst retention delta. The analysis script for this is not on the branch
yet (it belongs with the paper analysis in `paper_analysis/`).

How each possible outcome reads, pre-stated so the rebuttal commits to it before the
numbers arrive: table at the end of `docs/REBUTTAL_RUNBOOK.md`.

---

## 6. File-by-file

| file | change |
|---|---|
| `src/hivemind/controller/config.py` | `DebugConfig.policy_override` accepts `yoked_random`, `anti_commit` |
| `src/hivemind/controller/policy.py` | `_phase_salt`, `StoreAction.controller_store`, `_controller_label`, `_yoked_labels`, dispatch in `decide()` |
| `src/hivemind/training.py` | `trainable=routed` (freeze shared, digests, `shared_frozen_check` in run_summary, `trainable` in run_config/run_summary); `commits.jsonl` writer; `eval_rows.jsonl` sink; `eval.row_losses` into `EvaluationConfig` |
| `src/hivemind/evaluation.py` | `EvaluationConfig.row_losses`, `DomainMetrics.rows`, `_eval_one_domain(keep_rows=)`, `run_evaluation(row_sink=)` |
| `scripts/weight_drift.py` | `--expect-shared-unchanged` |
| `scripts/commit_schedule.py` | new: per-phase commits/defers, early fraction, decile histogram, mean C̄, yoked invariant + overlap, ratio vs reference, `--expect-match`, `--json` |
| `scripts/run_rebuttal.sh` | new: `check / smoke / launch / early / post / all` |
| `configs/experiments/rebuttal_v1.yaml` | new: the 21-run matrix |
| `configs/experiments/rebuttal_smoke.yaml` | new: the CPU smoke of the same arms |
| `tests/test_rebuttal_controls.py` | new, 24 tests (policy semantics, gates, `routed` through the real training loop, trace contents) |
| `tests/test_evaluation.py` | `test_row_sink_receives_every_row` |
| `docs/REBUTTAL_RUNBOOK.md`, `docs/REBUTTAL_CHANGES.md` | operator page; this page |
| `CLAUDE.md` | test count |

---

## 7. What was verified before the push, and what was not

Verified on 2026-11-08 (CPU box, `/root/hivemind`):

- 551/551 tests pass (2 GPU-only skips).
- Both manifests dry-run; the 21 resolved command lines carry the overrides of
  section 3; `stream_small` preflight (metadata) OK.
- CPU smoke 7/7 complete; the numbers quoted in section 2.1; `shared_unchanged=True`
  on all four frozen arms (19,448,448 shared coordinates on the smoke student).
- An independent read of the whole diff (second model, no shared context) found no
  defect in: R-count identity and `k <= candidates` in the yoked paths; CPU generator
  independent of rank/device and of the global RNG; freeze before DDP wrap and before
  the optimizer with the tied embedding/head as one `Parameter`; no gradient
  checkpointing path; traces main-rank only and keep-last under resume; every manifest
  key present or `++`-added; no `set -e` abort paths in the script.

Not verified, by construction of the box here (no GPU):

- That the frozen-shared arms learn and commit on the pretrained 0.6B (the `early`
  stage is the check).
- Throughput with the 256-row eval and three concurrent runs (the 12-16 h estimate is
  from the September campaign's per-run times plus the eval).
- The `trihope_e256` first-32-row M1 against 0.084 (GPU nondeterminism sigma from the
  paper: 0.005).

Not on this branch (paper side, `/root/trihope-paper`): the MoFO Random-BCD concession
in related work, softening "exact isolation" to what the ledger and digests actually
certify, the ppjp policy-level answer, presentation fixes, and the per-row bootstrap
script. Optional extras discussed and deferred: a MoLF-matched random arm, restoring the
`merge_delta` logging / `test_checksum_invariants.py` that REVISION_NOTES.md mentions
but no tree contains, and localising the 2-coordinate ledger discrepancy.

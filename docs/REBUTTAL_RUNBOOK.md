# ICLR-2027 rebuttal batch: operator runbook (2026-11-08)

> One manifest, one script. Read this page, then run `bash scripts/run_rebuttal.sh all`
> on the GPU box. Nothing here needs a decision from you; if something fails, stop and
> send the output.

Branch: **`rebuttal/iclr2027-controls`** (on top of `main` 1ed99a3, the commit the
submitted paper was built from). Base config, stream, student, teachers, thresholds
and seeds are the paper's. The only new things are three control arms, a trainability
mode, two small trace files and a bigger eval.

## What the reviewers asked for, and which arm answers it

All three reviews (ratings 2/2/2) say the same thing in different words: *the paper's
null result ("selection adds nothing measurable") is not yet shown to be
informative.* Two concrete objections carry it:

| Objection | Who | Arm(s) that answer it |
|---|---|---|
| The shared parameters (26 % of the model) train every step outside the gate and drift 65x further than the routed weights; training them alone already gives the full forgetting (M1 0.084 = 0.084). The routed commits may simply add nothing on top of that floor. | BDmh W1, 1Ume W3 | `shared_frozen_trihope`, `shared_frozen_yoked_random`, `shared_frozen_never_commit` (+ `shared_frozen_anti_commit`) |
| No positive control: nothing shows the benchmark *can* detect a selection effect. A deliberately bad selection at the controller's volume must forget more, otherwise the null only bounds the benchmark's resolution. | BDmh W2/Q2 | `anti_commit` (and `shared_frozen_anti_commit`) |
| The random control matched commit *rate* per phase, not commit *timing* (89 % of the controller's commits fall in the first 200 steps of a phase) or per-step count. Wanted: same per-step commit count, random module identity only. | 1Ume W1/Q1 | `yoked_random` (+ `commits.jsonl` and `scripts/commit_schedule.py`, which also produce the commit-time distribution asked for) |
| Retention measured on 32 rows per domain, no per-row uncertainty. | 1Ume W3/Q3, BDmh W2 | every arm here evaluates on 256 rows per domain and logs every row (`eval_rows.jsonl`); `trihope_e256` is the reference re-run under that eval |

`shared_frozen_never_commit` is BDmh's "ideally also never-commit".

## The runs

`configs/experiments/rebuttal_v1.yaml`: 7 arms x 3 seeds (1337, 2024, 7) = **21 live
runs** on `stream_small`, in priority order. Each run is ~1-2 h with three sharing an
A6000/A100; the 256-row eval adds ~10-15 min per run. Whole batch ~12-16 h at
`--concurrent 3`. If time runs short, the runner launches in manifest order, so the
later arms (`shared_frozen_anti_commit`, `shared_frozen_never_commit`) are the ones
that drop.

| # | arm | regime | override | what a reviewer reads off it |
|---|---|---|---|---|
| 1 | `anti_commit` | shared trainable | commit the k_t lowest-C̄ blocks | forgets more than controller/random -> benchmark resolves selection |
| 2 | `shared_frozen_trihope` | shared frozen | - | controller with the floor removed |
| 3 | `shared_frozen_yoked_random` | shared frozen | random k_t blocks | the comparison both reviewers asked for (vs 2) |
| 4 | `trihope_e256` | shared trainable | - | reference under the 256-row eval; must reproduce the paper's 0.084 on its first 32 rows |
| 5 | `yoked_random` | shared trainable | random k_t blocks | 1Ume's timing-matched control in the paper's regime |
| 6 | `shared_frozen_anti_commit` | shared frozen | lowest-C̄ k_t blocks | positive control in the most sensitive regime |
| 7 | `shared_frozen_never_commit` | shared frozen | P disabled | floor of the frozen regime |

**No run depends on another.** The yoked arms are *live-yoked*: on every step the
policy classifies exactly as the controller (same defers, same replay, same
confidence gate), counts its own commits k_t, and re-seats those k_t commits on other
non-deferred selected blocks. Count and timing match step for step within the arm's
own trajectory, so no share file from a previous run is needed and everything can run
concurrently. (The paper's `random_commit` needed the reference's per-phase shares;
the old `quota_shuffle` drifted to ~2x the commit volume. Neither problem exists here;
`commit_schedule.py` verifies it.)

## Steps

```bash
cd <the checkout that ran the live campaign>            # e.g. /home/a6000/asif/Council-TriHOPE
git fetch origin
git checkout -B rebuttal/iclr2027-controls origin/rebuttal/iclr2027-controls
pip install -e ".[dev,data]"                             # only if the env is new
```

```bash
bash scripts/run_rebuttal.sh check
```
~10 min, no GPU work: prints branch/commit, runs the unit tests for the new code paths
(must be all green), preflight, and the 21-row dry run. If the box-local
`stream_small.yaml` still carries the campaign's threshold edit that is fine: every
arm pins `surprise_high=1.25`, `repetition_low=0.40`, `replay_on_hit=true` itself.

```bash
bash scripts/run_rebuttal.sh smoke
```
~20 min on CPU (the GPU is free for `launch` meanwhile if you want to start it in a
second shell). Seven 70-step runs of every arm on the tiny stream. It prints one line
per arm; every line must end without `<-- FAIL`, every `shared_frozen_*` line must say
`shared_unchanged=True`, and the commit-schedule table must show `ctl` = `commits` on
the yoked arms. (Verified on 2026-11-08 on CPU: 7/7 arms finish; `trihope` 15 commits,
`yoked_random` 21, `anti_commit` 24, `ctl == commits` on both; the frozen arms commit 0
there for the reason given under `early` below.)

```bash
bash scripts/run_rebuttal.sh launch
```
Starts the 21 runs, three at a time, under `--live --resume`. Safe to interrupt and
re-run: finished runs are skipped, interrupted ones resume from their last checkpoint.
Use `CONCURRENT=2` on a 40 GB card. Progress: `runs/rebuttal_v1_live/*/status.json`,
`tail -f runs/rebuttal_v1_live/<run>/stdout.log`.

```bash
bash scripts/run_rebuttal.sh early          # in a second shell, about an hour after launch
```
One line per run: current step, loss at start -> now, committing steps so far. The
`shared_frozen_*` runs must show a falling loss, and `shared_frozen_trihope` must have
commits > 0 by the end of `code_recurrent` (step 1999). If a frozen arm's loss is flat,
stop and report. (In the CPU smoke the frozen arms never commit and their loss stays
at ~11: the tiny smoke student is randomly initialised, so freezing its embeddings
leaves nothing to learn with. That is expected there and says nothing about the 0.6B.)

```bash
bash scripts/run_rebuttal.sh post
```
After the runs (also fine on a partial batch):
1. `weight_drift.py --expect-shared-unchanged` on every `shared_frozen_*` run (exits
   non-zero if any shared coordinate moved; **report that rather than re-running**).
   Plain drift for the other arms.
2. `commit_schedule.py` per seed against `trihope_e256`: commit counts and the
   fraction of commits in the first 200 steps of each phase for every arm; the
   yoked-arm invariant (`ctl == commits`) and the overlap with the controller's picks.
3. `analysis.run_report` -> `runs/rebuttal_v1_live/analysis/`.
4. A tarball `rebuttal_results_<ts>.tgz` of the run directories without checkpoints.
   Send it, or push `runs/rebuttal_v1_live` (minus `checkpoints/`) to a results branch.

## What is new on this branch (for the record)

- `train.trainable=routed` (`src/hivemind/training.py`): freezes the `shared` group
  only (tied embedding/LM head, norms). `run_summary.json -> extra.shared_frozen_check`
  holds SHA-256 digests of every shared tensor before vs after the run and
  `unchanged: true/false`.
- `controller.debug.policy_override=yoked_random | anti_commit`
  (`src/hivemind/controller/policy.py::_yoked_labels`). Deterministic in
  `(train.seed, step, phase)`; bit-exact under resume; identical on every DDP rank.
- `commits.jsonl` next to `events.jsonl`: one record per step with R/F/P counts, and
  on committing steps the committed blocks, their C̄, and (yoked arms) the blocks the
  controller would have committed. ~0.5 MB per run. Kept with the results.
- `eval.row_losses=true` -> `eval_rows.jsonl`: every validation row's loss sum, token
  count and bucket per eval pass and domain. Enables a per-row bootstrap.
- `eval.max_batches=128` (manifest override): 256 rows per domain. Rows come in loader
  order, so the paper's 32 rows are the first 32 of each record.
- `scripts/commit_schedule.py`, `scripts/weight_drift.py --expect-shared-unchanged`,
  `scripts/run_rebuttal.sh`, `configs/experiments/rebuttal_v1.yaml`,
  `configs/experiments/rebuttal_smoke.yaml`, `tests/test_rebuttal_controls.py`.

Nothing in an existing arm's code path changed: the new override branches are only
reached with the new override values, and `trainable=all|lora` are untouched.

## Reading the results (for the authors, not the operator)

Metric: M1 as in the paper (worst retention delta), now on 256 rows, plus the 32-row
subset for continuity, plus in-phase (acquisition) loss and exact-match. Seed-paired
differences, 3 seeds, reported with the seed-level interval the paper already uses.

| Result | Meaning | What the rebuttal says |
|---|---|---|
| `anti_commit` forgets more than `trihope_e256` and `yoked_random` (3/3 seeds), controller ~ random | the benchmark resolves selection; the null is informative | "The gate throttles volume; its selection is no better than random, and we can now show the benchmark would have seen it." |
| `anti_commit` ~ controller ~ random | at ~74 commits per run selection is undetectable either way | retract "selection does not matter"; claim "at gated volume, permanent writes are too few to matter: the gate's benefit can only be throttling" |
| frozen: controller forgets less than yoked-random | selection matters once the floor is removed | the headline flips; the floor-masking becomes the main finding |
| frozen: controller forgets more than yoked-random | the gate selects harmfully | still publishable as an audit result; tone changes |
| frozen: every arm ~ 0 forgetting | nothing routed is enough to forget with | report acquisition loss alongside; the frozen regime is a floor at the other end |
| `trihope_e256` 32-row M1 != 0.084 +- sigma_GPU | code/box drift | report it with the GPU-nondeterminism sigma from the paper (0.005) |

Pre-state these readings in the rebuttal **before** the numbers arrive.

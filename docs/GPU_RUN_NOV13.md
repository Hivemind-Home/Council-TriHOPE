# GPU run sheet: ICLR-2027 rebuttal, batches of 2026-11-13

> **For the GPU operator.** Follow this page top to bottom. Every command is here; nothing
> needs a decision from you except where a box says **STOP**. If anything looks different
> from what is described, stop and send the output. The rebuttal deadline is close, so
> the order matters: the first 6 runs are the ones the paper needs most.

Same box and checkout as the batch-1 runs (`runs/rebuttal_v1_live` must be there).
Branch: **`rebuttal/extra-seeds`**. One script does everything:
`scripts/run_rebuttal_nov13.sh {check | run | smoke | early | post}`.
Every stage writes its console output to `runs/rebuttal_nov13_logs/`.

---

## Timeline at a glance

| When | Shell 1 | Shell 2 |
|---|---|---|
| t = 0 | `check` (~5 min) | |
| t ≈ 5 min | `run` (≈ 12-13 h in total, keep it running) | `smoke` (CPU, ~15 min) |
| t ≈ 45 min | | `early` |
| t ≈ 4 h | (step 1 done: 6 frozen oracle runs) | `post` → **send the tarball right away** |
| t ≈ 9 h | (step 2 done: extra seeds) | `post` → send |
| t ≈ 13 h | (step 3 done: everything) | `post` → send |

Measured on batch 1: one run takes 1.6-1.8 h with three sharing the GPU.

---

## 0. Get the code

Stop any batch that is still running from earlier instructions (Ctrl-C once in its
shell; every run saves a checkpoint). Its interrupted runs are finished later by `run`:
`check` (step 1) repairs the status of runs that the previous version of the runner
wrongly marked `done` after a Ctrl-C, and prints each one it repairs. Then:

```bash
cd <the checkout that ran batch 1>        # the one with runs/rebuttal_v1_live in it
git fetch origin
git checkout -B rebuttal/extra-seeds origin/rebuttal/extra-seeds
pip install -e ".[dev,data]"               # only if the environment is new
```

Make sure nothing else is using the GPU: `nvidia-smi` should show no other processes.

## 1. Check (shell 1, ~5 min, no GPU training)

```bash
bash scripts/run_rebuttal_nov13.sh check
```

What it does: prints branch and commit, GPU name and memory, repairs the status of
interrupted runs (see step 0), verifies the six batch-1 reference runs the new arms read,
runs the unit tests for the new code, preflight, and dry-runs the 20 runs.

**Good:** the last line is `CHECK OK - launch with: ...`.
The reference list shows six `ok` lines (`shared_frozen_trihope` and `trihope_e256`, seeds
1337 / 2024 / 7). The dry-runs list 12 `rebuttal_oracle_live` runs + 4 + 4 seed runs.

> **STOP** if it prints `CHECK FAILED`, or any reference says `MISSING/NOT DONE`.
> Send `runs/rebuttal_nov13_logs/check_*.log`.

## 2. Run (shell 1, keep it open; ≈ 12-13 h)

```bash
bash scripts/run_rebuttal_nov13.sh run
```

It runs three steps in this order, three runs at a time on the GPU:

| Step | What | Runs | Time | Output dir |
|---|---|---|---|---|
| 1 | interference-oracle bracket, shared-frozen | 6 | ~3.5-4 h | `runs/rebuttal_oracle_live/shared_frozen_*` |
| 2 | extra seeds 42 and 314, shared-frozen | 8 | ~5 h | `runs/rebuttal_seeds_live`, `runs/rebuttal_seeds_sched_live` |
| 3 | interference-oracle bracket, shared-trainable | 6 | ~3.5-4 h | `runs/rebuttal_oracle_live/schedule_*` |

In step 2 the script starts the second half (stage B) by itself as soon as both
`shared_frozen_trihope-seed42/314` runs are finished. Do not start anything by hand.

**If it stops for any reason** (crash, Ctrl-C, reboot): run the same command again.
Finished runs are skipped; interrupted runs continue from their last checkpoint and end
bit-identical to an uninterrupted run (tested). An interrupted run is never counted as
finished, and an arm whose reference run is unfinished refuses to start, so a stop
cannot silently produce a truncated result.

**If you see `CUDA out of memory`:** Ctrl-C, then
```bash
CONCURRENT=2 bash scripts/run_rebuttal_nov13.sh run
```

## 3. Smoke (shell 2, right after launching `run`, CPU, ~15 min)

```bash
bash scripts/run_rebuttal_nov13.sh smoke
```

Two 70-step runs of the new arms on the tiny CPU stream (it does not use the GPU).
**Good:** `manipulation check: PASS`, and two `TOTAL` lines whose last number is `1.00`.

> **STOP the GPU run** (Ctrl-C in shell 1) if it prints `FAIL` or a Python traceback,
> and send `runs/rebuttal_nov13_logs/smoke_*.log`.

## 4. Early check (shell 2, about 45 min after `run` started)

```bash
bash scripts/run_rebuttal_nov13.sh early
```

One line per started run: step, loss at start → now, committing steps so far.

**Good:**
- every running run shows a falling loss and a step count that moves between two calls;
- no line ends in `<-- CUDA OOM`;
- the oracle table: runs past step ~500 show `scored` > 0 and `right side` ≥ 0.90.
  Before step 500 every commit is a fallback (no earlier domain exists yet), so
  `scored` = 0 and `FAIL` is expected **at that point**.

> **STOP and report** if a run has `scored` > 10 and `right side` < 0.90, if a loss is
> flat or `nan`, or on OOM (then restart with `CONCURRENT=2`, step 2 above).

You can run `early` again at any time; it only reads.

## 5. Post (shell 2): after step 1, again after step 2, again at the end

```bash
bash scripts/run_rebuttal_nov13.sh post
```

Works on a partial batch. It writes everything to `runs/rebuttal_nov13_analysis/` and
makes `rebuttal_nov13_results_<time>.tgz` (no checkpoints, no `events.jsonl`).

What it checks (all must hold; report any that does not, **do not re-run anything**):
1. `weight_drift`: every frozen run has its shared parameters unchanged (`ok`).
2. `oracle_check`: `manipulation check: PASS`.
3. `commit_schedule`: every `TOTAL` line ends in `1.00` and no `FAIL`.
4. `m1_rows`: `validation mismatches: 0`, followed by the results table.

**Send the tarball as soon as step 1 has finished** (`status.txt` shows the six
`shared_frozen_schedule_oracle_*` runs as `done`). Then again after step 2 and at the end.

---

## What these runs are (for context, not needed to operate)

Reviewer BDmh asked for a deliberately bad selection at the controller's commit volume:
"committing the selected blocks with the lowest agreement C̄, **or those with the largest
interference with earlier domains**". The lowest-C̄ version ran in the volume batch and did
not forget more. These runs do the interference version:

- `schedule_oracle_bad` / `schedule_oracle_good` commit exactly as many blocks per step as
  the batch-1 controller run of the same seed, choosing the blocks whose update this step
  is predicted to hurt earlier domains the **most** / the **least**. The prediction is the
  dot product of the exact optimizer update with the gradient on 4 training rows of every
  other domain (`src/hivemind/controller/oracle.py`). In the first phase there is no earlier
  domain, so both fall back to a random pick (marked `oracle_fallback` in `commits.jsonl`).
- The extra seeds (42, 314) repeat the shared-frozen controller, never-commit,
  volume-matched random and lowest-C̄ arms, so those comparisons rest on 5 seeds, not 3.

Checks done before handing over (2026-11-13, CPU): 572 unit tests pass, including:
- the predicted update equals MaskedAdamW's actual step;
- each score equals the measured loss change within 5%;
- scoring leaves gradients, optimizer state and RNG untouched;
- an oracle run interrupted at step 52 and resumed reproduces the uninterrupted run
  exactly (every commit, oracle score and eval-row loss).
CPU smoke: exact volume match, manipulation check PASS. An independent code review
found one real defect, now fixed: the old runner marked a Ctrl-C'd run `done` under
`--concurrent`. All 33 batch-1 runs were checked: none was affected (no interrupted
flag, all 6000 steps).

Per-run peak GPU memory in batch 1 was ~29 GB. The oracle step runs after the main
backward on a single row, so it should not raise that peak.

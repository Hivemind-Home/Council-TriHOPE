# GPU timeline: final ICLR batch, Wed 23 Sep → Thu 24 Sep 2026

For the GPU operator. It lists everything that still has to run, in order, with the time it should
happen and the exact command. Background and rationale: `docs/FINAL_BATCH_RUNBOOK.md` (same branch).

All times are **UTC / Bangladesh time (UTC+6)**. The schedule assumes launch at **15:30 UTC
(21:30 BDT) Wed**. If you launch earlier or later, shift every row by the same amount; nothing else
changes. If you already started, find the row you are on and continue from there.

## Deadlines

| When (UTC / BDT) | What |
|---|---|
| **Wed 17:30 / Wed 23:30** | Latest launch that still finishes the full batch (about 14 h, 17 h with slack) before the freeze. Launching later: cut runs (see "Running late"). |
| **Thu 08:00 / Thu 14:00** | Target: all runs done, reductions pushed. |
| **Thu 12:00 / Thu 18:00** | **Data freeze for the main text.** Anything later goes to the appendix only. |
| Fri 12:00 / Fri 18:00 | Appendix freeze. Nothing after this is used. |
| Sat 11:59 / Sat 17:59 | Paper deadline (Fri 25 Sep AoE). |

## The list: 34 runs, about 14 h

The runner goes spec by spec, then seed by seed, and keeps 3 runs going at once (a new one starts as
soon as one finishes). With 3 sharing the GPU, one run takes 0.75–2 h, as measured in the earlier live
campaign (`topm_all` is the slowest, about 2 h). The "Group" column is the approximate order.

| Group | Runs | Manifest | Why it matters | If late |
|---|---|---|---|---|
| 1 | `frozen_blocks` seeds 1337, 2024, 7 | `priority_s3` | Only the shared params train. **Decides the paper's framing (DP1)**, and checks Theorem 1 at scale (routed weights bit-identical). | never cut |
| 2 | `trihope_sentinel` 1337, 2024, 7 | `priority_s3` | Same as E1 TriHOPE. Detects drift; grows the reference pool to 10 (DP2). | keep ≥1 |
| 3 | `periodic_merge` 1337, 2024, 7 | `priority_s3` | LoRA + merge every 250 steps. A merge baseline that actually fires. | cut to 1 seed (last resort) |
| 4 | `random_commit` 1337, 2024, 7 | `priority_s4` | **Tests the headline claim (H1b, DP4):** random commits with TriHOPE's own defer/replay. | never cut |
| 5 | `surprise_gate_s4` 2024, 7; `molf_style_a0p7` 2024 | `priority_s2` | Strongest competitors to n=3. | never cut |
| 6 | `molf_style_a0p7` 7; `no_cosine` 2024, 7 | `priority_s2` | Competitor + the gate ablation (H3) to n=3 (DP3). | never cut |
| 7 | `molf_style_a0p5` 2024, 7; `moments_optimizer` 2024 | `priority_s2` | SNR rule (H4) to n=3; tracker ablation. | cut 7th / 5th |
| 8 | `moments_optimizer` 7; `no_surprise` 2024, 7 | `priority_s2` | Ablations to n=3. | cut 5th / 4th |
| 9 | `topm_2` 2024, 7; `topk_25` 2024 | `priority_s2` | Ablations to n=3. | cut 3rd |
| 10 | `topk_25` 7; `topm_all` 2024, 7 | `priority_s2` | Ablations to n=3. | cut 3rd / 2nd |
| 11 | `pgate_c0p35`, `pgate_c0p65`, `pgate_c0p2` (seed 1337) | `priority_s1` | Dose-response of the C̄ permanence gate. | cut 6th (keep c0p35) |
| 12 | `pgate_c0p8` (seed 1337) | `priority_s1` | Top of the dose-response. | cut 1st |

Output directories: `runs/priority_s{1,2,3,4}_v1_live/<run>-seed<seed>/`.

## Timeline with commands

### ☐ 15:00 UTC / 21:00 BDT: record what ran before, then get the batch (15 min)

Run this **before switching branches**, in the checkout that ran the 115 live runs. Don't stash first:
the local `stream_small.yaml` edit is part of what gets recorded.

```bash
cd /home/a6000/asif/Council-TriHOPE
git fetch origin +refs/heads/campaign/iclr-2027-final-batch:refs/remotes/origin/campaign/iclr-2027-final-batch
git show origin/campaign/iclr-2027-final-batch:scripts/provenance_snapshot.sh | bash
git checkout -B campaign/iclr-2027-final-batch origin/campaign/iclr-2027-final-batch
git log --oneline -1          # must show the newest commit of this branch on GitHub
git status --short            # expect: M configs/stream_small.yaml (the local threshold edit) and ?? provenance/
```

These forms work on any clone, including a single-branch one. (A plain `git checkout <branch>`
fails there; tested.) **If you already switched to this branch earlier and have no local commits,
run `git pull` instead of the checkout line.** `-B` resets the local branch to GitHub's copy.

Disk and tools. Each run keeps its final checkpoint (about 6 GB: weights, optimizer state,
controller state), so the batch needs **about 250 GB free** (34 × 6 GB, plus 3 runs in flight):

```bash
df -h .
command -v rsync tmux        # both must print a path
```

If there is less than 250 GB free, free space from **finished** runs only. This deletes the
optimizer, controller and store state from their checkpoints (about 4.5 GB each). It keeps
`base.pt` and `lora.pt`, which is all the weight-drift check needs. It never touches a run that is
not `done`, because a crashed run may need that state to resume:

```bash
for r in runs/*_live/*-seed*; do
  grep -q '"state": "done"' "$r/status.json" 2>/dev/null || continue
  rm -f "$r"/checkpoints/step_*/{optimizer,controller,stores}.pt
done
df -h .
```

`random_commit` reads TriHOPE's per-phase decision shares from the E1 report. Make sure the file is
there (if it's missing, the 3 `random_commit` runs stop with an error):

```bash
ls runs/baselines_small_v1_live/analysis/action_share_by_phase.csv || \
  { mkdir -p runs/baselines_small_v1_live/analysis && \
    cp results_live/baselines_small_v1_live/analysis/action_share_by_phase.csv runs/baselines_small_v1_live/analysis/; }
```

1-step dry run of every manifest. Each should list its runs as `pending` with no error:

```bash
for m in priority_s3 priority_s4 priority_s2 priority_s1; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --live --dry-run | awk '{print $1, $2}'; done
```

### ☐ 15:30 UTC / 21:30 BDT: launch the whole chain in tmux

It runs about 14 h unattended, so start it in tmux, where it survives an SSH disconnect. The four
manifests are joined with `;`, **not** `&&`. The runner exits with an error if any single run
fails, and with `&&` one crash would stop every later manifest overnight.

```bash
tmux new -s batch
cd /home/a6000/asif/Council-TriHOPE
mkdir -p logs
( python scripts/run_experiment.py configs/experiments/priority_s3.yaml --live --resume --concurrent 3 ; \
  python scripts/run_experiment.py configs/experiments/priority_s4.yaml --live --resume --concurrent 3 ; \
  python scripts/run_experiment.py configs/experiments/priority_s2.yaml --live --resume --concurrent 3 ; \
  python scripts/run_experiment.py configs/experiments/priority_s1.yaml --live --resume --concurrent 3 ) 2>&1 | tee -a logs/final_batch.log
# detach: Ctrl-b then d      reattach: tmux attach -t batch
```

### ☐ 15:35 UTC / 21:35 BDT: drift check (2 min after launch)

In a second tmux window (`Ctrl-b c`):

```bash
python scripts/check_config_drift.py runs/priority_s3_v1_live/*-seed* --committed runs
```

Every line must say `ok`. **Any `[DRIFT]` line: stop the chain (Ctrl-c in window 0) and send the output.**

### ☐ ~16:30 UTC / 22:30 BDT: frozen_blocks ×3 done: push (DP1)

Define the push helper once per shell. It copies only small files, skips runs still in progress, and
doesn't pull code:

```bash
push_results () {
  rsync -a --exclude events.jsonl --exclude checkpoints \
        $(ls -d runs/priority_s{1,2,3,4}_v1_live 2>/dev/null) results_live/
  git add results_live
  find results_live -name '*.log' -size -20M -print0 | xargs -0 -r git add -f   # *.log is gitignored
  [ -d provenance ] && git add provenance
  git commit -q -m "final batch: $1"
  git pull -q --rebase --autostash origin campaign/iclr-2027-final-batch && \
    git push -q origin campaign/iclr-2027-final-batch
}
push_results "frozen_blocks x3"
git log origin/campaign/iclr-2027-final-batch --oneline -1   # must show the commit you just made
```

Three things this helper handles (all tested):
- `git add` of a missing `provenance/` would make the whole add fail, so nothing would be pushed.
- `*.log` is gitignored, so without `-f` the runs' `stdout.log` and `train.log` (what we need to
  debug a crash) would be left out.
- It rebases onto GitHub before pushing. If the lead pushes a script fix during the night, the
  auto-push still goes through instead of being rejected every time. Your local `stream_small.yaml`
  edit is kept. Nothing under `src/` or `configs/` will change during the batch.

"nothing to commit" is harmless; a push that failed earlier is retried on the next call.

Then run the Theorem-1 bit-identity check on the frozen runs (CPU, loads Qwen3-0.6B once):

```bash
python scripts/weight_drift.py runs/priority_s3_v1_live/frozen_blocks-seed* --expect-routed-unchanged
```

It must report the routed weights unchanged for all 3 seeds. If it doesn't, send the output.

### ☐ 16:45–18:00 UTC / 22:45–00:00 BDT: CPU reductions on the existing live runs (while the GPU keeps going)

```bash
# T2: decision counts by action x surface x phase (the 115 earlier live runs; the new ones are done in the morning)
python scripts/events_decision_counts.py $(ls -d runs/*_live/*-seed* | grep -v priority_)

# T3: weight drift split routed / adapters / shared for the key E1 runs
python scripts/weight_drift.py runs/baselines_small_v1_live/{trihope,surprise_gate,no_consolidation,random_routing,full_ft,molf_style}-seed1337 \
    runs/r_tier_small_v1_live/p_only-seed1337 runs/budget_sweep_small_v1_live/molf_style_a0p7-seed1337
```

### ☐ Before sleeping (~00:00 BDT): start the auto-push loop

It pushes every 2 hours, so results reach the paper overnight. Run it in a third tmux window
(`Ctrl-b c`), after pasting the `push_results` definition from above into that window too:

```bash
cd /home/a6000/asif/Council-TriHOPE
while true; do push_results "auto $(date -u +%H:%M)"; sleep 7200; done
```

Check in the morning with `git log origin/campaign/iclr-2027-final-batch --oneline -3` that the auto-pushes went through.

### Overnight (runs unattended; times approximate, ±1 h)

| Done by (UTC) | Done by (BDT) | Runs finished |
|---|---|---|
| Wed 16:30 | Wed 22:30 | `frozen_blocks` ×3 (DP1) |
| Wed 17:40 | Wed 23:40 | `trihope_sentinel` ×3 (DP2) |
| Wed 18:25 | Thu 00:25 | `periodic_merge` ×3 (end of `priority_s3`) |
| Wed 19:40 | Thu 01:40 | `random_commit` ×3 (DP4; end of `priority_s4`) |
| Wed 21:50 | Thu 03:50 | `surprise_gate_s4`, `molf_style_a0p7`, `no_cosine` ×2 each (DP3) |
| Wed 23:50 | Thu 05:50 | `molf_style_a0p5`, `moments_optimizer`, `no_surprise` ×2 each |
| Thu 03:25 | Thu 09:25 | `topm_2`, `topk_25`, `topm_all` ×2 each (end of `priority_s2`) |
| Thu 05:40 | Thu 11:40 | `pgate_c0p35`, `pgate_c0p65`, `pgate_c0p2`, `pgate_c0p8` (end of `priority_s1`) |

### ☐ Thu morning, when you wake: check progress and the later drift checks

```bash
for m in priority_s3 priority_s4 priority_s2 priority_s1; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --live --dry-run | awk '{print $1, $2}'; done
python scripts/check_config_drift.py runs/priority_s4_v1_live/*-seed* --committed runs
python scripts/check_config_drift.py runs/priority_s2_v1_live/*-seed* --committed runs
python scripts/check_config_drift.py runs/priority_s1_v1_live/*-seed* --committed runs   # once s1 has started
```

Any `failed` run: see "If something goes wrong". Any `[DRIFT]` line: send the output.

If the batch was started with an older copy of this doc (manifests joined with `&&`) and a run
failed, the later manifests never started. Check `tail logs/final_batch.log`, and relaunch the
remaining manifests with the `;` chain above (`--resume` skips everything already done).

### ☐ ~06:00–08:00 UTC / 12:00–14:00 BDT: after the last run, final reductions and final push

Stop the auto-push loop (Ctrl-c in its window) first.

```bash
# T3 on the new sentinel and periodic-merge runs
python scripts/weight_drift.py runs/priority_s3_v1_live/{trihope_sentinel,periodic_merge}-seed1337

# T2 on the new runs
python scripts/events_decision_counts.py runs/priority_s*_v1_live/*-seed*

# T4: regenerate every live report
for m in baselines_small_v1_live bad_teacher_small_v1_live r_tier_small_v1_live p_study_small_v1_live \
         budget_sweep_small_v1_live ablation_grid_v1_live priority_s3_v1_live priority_s4_v1_live \
         priority_s2_v1_live priority_s1_v1_live; do
  python -m analysis.run_report runs/$m; done

# copy results + small reductions (never events.jsonl or checkpoints), commit, push
rsync -a --exclude events.jsonl --exclude checkpoints runs/priority_s{1,2,3,4}_v1_live results_live/
for d in runs/*_live; do
  rsync -am --include '*/' --include 'analysis_t2/**' --include 'weight_drift.json' --include 'analysis/**' \
        --exclude '*' "$d" results_live/; done
git add results_live
find results_live -name '*.log' -size -20M -print0 | xargs -0 -r git add -f
[ -d provenance ] && git add provenance
git commit -m "final batch: all priority runs, T2/T3/T4 reductions, provenance"
git pull --rebase --autostash origin campaign/iclr-2027-final-batch
git push origin campaign/iclr-2027-final-batch
git log origin/campaign/iclr-2027-final-batch --oneline -1   # must show this commit
```

**Must be pushed before Thu 12:00 UTC / 18:00 BDT.**

## If something goes wrong

**A run crashed.** Restart it from scratch with identical settings. Never resume a crashed run with
a changed override; that is what invalidated live `gradient_routing`.

```bash
rm -rf runs/priority_s2_v1_live/no_cosine-seed7          # example: the crashed run's dir
python scripts/run_experiment.py configs/experiments/priority_s2.yaml --live --resume --only no_cosine --concurrent 3
```

`--resume` skips the seeds that are already done, so only the deleted one reruns.

**A run diverged** (loss NaN or exploding). Keep it and report it; don't delete it.

**Drift check fails.** Stop the chain and send the output. Don't run more until it is resolved.

**Running late.** Relaunch the remaining manifests with `--skip`, cutting in this order:
1. `pgate_c0p8`
2. `topm_all`
3. `topk_25`, `topm_2`
4. `no_surprise`
5. `moments_optimizer`
6. `pgate_c0p65`, `pgate_c0p2`
7. `molf_style_a0p5`

For example:

```bash
python scripts/run_experiment.py configs/experiments/priority_s2.yaml --live --resume --concurrent 3 --skip topm_all,topk_25,topm_2
python scripts/run_experiment.py configs/experiments/priority_s1.yaml --live --resume --concurrent 3 --skip pgate_c0p8
```

**Never cut:** `frozen_blocks` ×3, `random_commit` ×3, `trihope_sentinel` (at least 1),
`surprise_gate_s4`, `molf_style_a0p7`, `no_cosine`.

## Not in this batch

This is the final GPU batch. A design-faithful variant, where a withheld base block writes its
paired LoRA adapter as in `docs/theory_101.md`, and a TriHOPE arm with the shared parameters frozen
are **not** run. The paper reports both as limitations.

## What happens on the paper side after each push

The lead pulls this branch, runs the config-drift check, regenerates every table and figure
(`make data results figures`), and applies the pre-registered decision rules (DP1–DP4). No numbers
need to be sent by hand; pushing the results is enough.

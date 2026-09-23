# Final batch for the ICLR-2027 paper: operator runbook (2026-09-23)

> **Follow `docs/GPU_TIMELINE_2026-09-23.md` for the order, times and commands.** This runbook
> keeps the background. Where the two differ, the timeline doc is correct.

Paper deadline: **Sat Sep 26, 11:59 UTC** (Fri Sep 25 AoE). Data for the main text is frozen **Thu Sep 24, 12:00 UTC**. Runs that land after that go to the appendix, until Fri 12:00 UTC.

Everything below runs on **the same box and checkout that ran the 115 live runs**, under `--live`, with `--concurrent 3`. One run takes 0.75–2 h with 3 sharing the GPU. There are 34 runs in total, about 14 h.

## 0. Before switching branches (5 min): record what ran

```bash
cd /home/a6000/asif/Council-TriHOPE           # the checkout that ran the live campaign
git fetch origin +refs/heads/campaign/iclr-2027-final-batch:refs/remotes/origin/campaign/iclr-2027-final-batch
git show origin/campaign/iclr-2027-final-batch:scripts/provenance_snapshot.sh | bash
```

This writes `provenance/<ts>/` from the *current* working tree: git HEAD + diff, the box-local
`stream_small.yaml` and its `.bak` files, `pip freeze`, torch/CUDA/device. Do not stash first:
the local threshold edit is part of what must be recorded. It is committed in step 5.

## 1. Get the batch

```bash
git checkout -B campaign/iclr-2027-final-batch origin/campaign/iclr-2027-final-batch   # keeps your local stream_small.yaml edit
```

The new manifests **pin** `surprise_high=1.25` and `repetition_low=0.40` on every spec. The resolved config is therefore the live one whether or not the local edit survives. Step 3 checks this.

## 2. Wave 0: smoke tests (≤15 min, optional if pressed)

These were already run on CPU with `stream_smoke` (70 steps):
- `frozen_blocks` opens 0 routed coordinates, makes 0 merges and 0 replays.
- `periodic_merge` merges at every window boundary.
- `pgate_*` resolves its thresholds.

On the box, a 1-step dry run of each manifest is enough:

```bash
for m in priority_s3 priority_s4 priority_s2 priority_s1; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --live --dry-run; done
```

## 3. Launch (in this order, one after the other)

```bash
python scripts/run_experiment.py configs/experiments/priority_s3.yaml --live --resume --concurrent 3 ; \
python scripts/run_experiment.py configs/experiments/priority_s4.yaml --live --resume --concurrent 3 ; \
python scripts/run_experiment.py configs/experiments/priority_s2.yaml --live --resume --concurrent 3 ; \
python scripts/run_experiment.py configs/experiments/priority_s1.yaml --live --resume --concurrent 3
# joined with ';' not '&&': the runner exits non-zero if any run fails, and '&&' would then skip every later manifest
```

> **Added after the first push (2026-09-23, `priority_s4`):** if you already launched
> `priority_s3`, just `git pull` on this branch (running jobs are unaffected: the only code
> change is a new `policy_override=random_commit` branch that no other arm reaches) and run
> `priority_s4` right after `priority_s3` finishes, before `priority_s2`. It needs
> `runs/baselines_small_v1_live/analysis/action_share_by_phase.csv`, which the live E1 report
> already wrote.

**Within 2 minutes of the first launch**, as soon as each run's `.hydra/config.yaml` exists, check drift:

```bash
python scripts/check_config_drift.py runs/priority_s3_v1_live/*-seed* --committed runs
```

Every line must say `ok`. Any `[DRIFT]` line means the resolved config differs from the committed live runs on a key the spec did not set. **Stop the batch and send the output.** Repeat the check for `priority_s2_v1_live` and `priority_s1_v1_live` when they start.

| manifest | runs | what it answers |
|---|---|---|
| `priority_s3` (3 seeds) | `frozen_blocks` | Only the unrouted (shared) params train. How much of forgetting and learning comes from them alone? **Decides the paper's framing.** |
| | `trihope_sentinel` | Identical to E1 `trihope`. Detects drift and grows the reference pool to 10 runs. |
| | `periodic_merge` | LoRA plus merge-and-reset every 250 steps. A merge-schedule baseline that actually fires; `plateau_trigger` never did. |
| `priority_s4` (3 seeds) | `random_commit` | TriHOPE's own defer/replay decisions, commits reassigned at random at TriHOPE's per-phase commit share. Isolates *commit selection* from replay volume; `random_routing` confounds the two (it replays ~4.7x more rows). **Tests the paper's headline claim.** |
| `priority_s2` (seeds 2024, 7) | `surprise_gate_s4`, `molf_style_a0p7`, `no_cosine`, `molf_style_a0p5`, `moments_optimizer`, `no_surprise`, `topm_2`, `topk_25`, `topm_all` | Brings the strongest competitors and the load-bearing ablations to n=3. |
| `priority_s1` (seed 1337) | `pgate_c0p35`, `pgate_c0p65`, `pgate_c0p2`, `pgate_c0p8` | Dose-response of the C̄ permanence gate. |

**If time runs short, cut from the bottom** in this order:
1. `pgate_c0p8`
2. `topm_all`
3. `topk_25`, `topm_2`
4. `no_surprise`
5. `moments_optimizer`
6. the rest of `pgate` (keep `c0p35`)
7. `molf_style_a0p5`

**Never cut:** `frozen_blocks` ×3, `random_commit` ×3, `trihope_sentinel` ×1 or more, `surprise_gate_s4`, `molf_style_a0p7`, `no_cosine`.

**Rules:**
- A crashed run is **restarted from scratch with identical overrides**. Delete its dir and rerun with `--only <id>`. Never resume it with a changed override; that is what invalidated live `gradient_routing`.
- A diverged run is reported, not deleted.

## 4. CPU tasks while the GPU runs

The event logs and checkpoints exist only on the box. These reduce them to small files for the paper.

```bash
# T2: decisions by action x surface (base block vs adapter) x phase; effective vs nominal merges
python scripts/events_decision_counts.py runs/*_live/*-seed*

# T3: weight change split routed-base / adapters / shared, from the final checkpoints.
# Theorem-1 check at scale: routed weights must be bit-identical in frozen_blocks.
python scripts/weight_drift.py runs/priority_s3_v1_live/frozen_blocks-seed* --expect-routed-unchanged
python scripts/weight_drift.py runs/baselines_small_v1_live/{trihope,surprise_gate,no_consolidation,random_routing,full_ft,molf_style}-seed1337 \
    runs/r_tier_small_v1_live/p_only-seed1337 runs/budget_sweep_small_v1_live/molf_style_a0p7-seed1337 \
    runs/priority_s3_v1_live/{trihope_sentinel,periodic_merge}-seed1337

# T4: regenerate every live report (needs events.jsonl) once the batch is done
for m in baselines_small_v1_live bad_teacher_small_v1_live r_tier_small_v1_live p_study_small_v1_live \
         budget_sweep_small_v1_live ablation_grid_v1_live priority_s3_v1_live priority_s2_v1_live priority_s1_v1_live; do
  python -m analysis.run_report runs/$m; done
```

`weight_drift.py` loads `Qwen/Qwen3-0.6B` once, on CPU.

## 5. What to send back (commit and push to this branch)

Copy into `results_live/` exactly as the first live campaign did: `run_summary.json`, `metrics.jsonl`, `status.json`, `stdout.log` and `hydra/`. Do **not** copy `events.jsonl` or `checkpoints/`. Then:

```bash
rsync -a --exclude events.jsonl --exclude checkpoints runs/priority_s{1,2,3,4}_v1_live results_live/
for d in runs/*_live; do   # T2/T3/T4 reductions of every live matrix (small files only)
  rsync -am --include '*/' --include 'analysis_t2/**' --include 'weight_drift.json' --include 'analysis/**' \
        --exclude '*' "$d" results_live/; done
git add results_live
find results_live -name '*.log' -size -20M -print0 | xargs -0 -r git add -f   # *.log is gitignored
[ -d provenance ] && git add provenance
git commit -m "final batch: priority runs, T2/T3/T4 reductions, provenance"
git pull --rebase --autostash origin campaign/iclr-2027-final-batch && git push origin campaign/iclr-2027-final-batch
```

Push **as waves finish**, not only at the end. The paper pipeline picks up new seeds automatically.

**Decision points** (send a one-line note when each is reached):
- **DP1**, after `frozen_blocks` ×3: its M1 and final macro loss.
- **DP2**, after `trihope_sentinel`: whether it lands inside 0.0838 ± 0.012 (M1).
- **DP3**, after `surprise_gate_s4` and `molf_style_a0p7` ×2.
- **DP4**, after `random_commit` ×3: its M1, final loss and replayed rows.

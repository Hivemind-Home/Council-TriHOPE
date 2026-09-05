# TriHOPE training pipeline and test commands — 2026-09-04

Branch `iclr2027-push` (12 commits on top of `main` @ `8dc4917`).
Everything below was verified on 2026-09-04 on a CPU box against real
Hugging Face Layer-C data via `stream_smoke`; the GPU campaign itself has
not been run yet. Machine assumed: one 96 GB GPU, cache-mode teachers (no
teacher model is loaded — the only model in memory is the Qwen3-0.6B
student). Companion docs: `docs/GPU_RUNBOOK.md` (tiers and budgets),
`docs/STATUS.md` (result slots), `docs/CHANGELOG_iclr.md` (every change),
`docs/running.md` (all knobs).

---

## 0. Before leaving the dev box

The branch lives only on the dev machine until it is pushed:

```bash
git push -u origin iclr2027-push        # on the dev box
```

## 1. Setup (once)

```bash
git fetch origin && git checkout iclr2027-push
git log --oneline -1                     # expect the "docs: dated training-pipeline…" commit or later
python -m venv .venv && source .venv/bin/activate      # or your env
pip install -e ".[dev,data,analysis]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

Datasets download automatically from the Hugging Face hub on first use
(`hivemind-research/{general,code,math,medical}-layerC-200k`; code is
~870 MB). Set `HF_HOME` if the default cache is on a small disk.

## 2. Tests (before anything touches the GPU)

```bash
python -m pytest tests/ -q                     # expected: 512 passed, 0 skipped (2 of them GPU-only)
ruff check src tests analysis scripts          # expected: All checks passed!
```

If `pip install` fails on `bitsandbytes` or `kernels` (both only needed
for live teachers / FP8 checkpoints), install without them:
`pip install -e ".[dev,analysis]" datasets transformers huggingface_hub pyarrow accelerate`.

Test groups worth knowing:

| Command | What it proves |
|---|---|
| `pytest tests/test_masked_adamw.py tests/test_ledger_completeness.py -q` | Theorem 1 / Corollary 3: closed coordinates never move; every change is a logged write |
| `pytest tests/test_resume_exact.py -q` | bit-exact resume, including after an R-store replay |
| `pytest tests/test_replay.py tests/test_baseline_policies.py -q` | replay trigger + the four E1 baseline controllers |
| `pytest tests/test_corrupt_teacher.py tests/test_attribution.py tests/test_rollback.py tests/test_gradient_routing.py -q` | the whole E5 chain |
| `pytest tests/test_distributed_equivalence.py tests/test_distributed_replay.py -q` | two-rank gloo: identical decisions, replay row + teacher index synced |
| `pytest tests/test_budget_curve.py tests/test_analysis.py -q` | Figure 1 tables and the report |

## 3. Smoke checks on real data (3–5 min each; they run on CPU by design)

`stream_smoke` pins `train.device: cpu` so the reference losses below are
reproducible on any machine — do not move it to CUDA. Each command gets
its own directory under `runs/smoke/`: the event log is append-mode, so
runs that share a directory would mix their traces.

```bash
# 3a. baseline behaviour — must end at EXACTLY this loss (reference from docs/experiments.md)
python train.py --config-name stream_smoke \
    ++run.dir=runs/smoke/plain ++checkpoint.dir=runs/smoke/plain/ckpt \
    ++logging.path=runs/smoke/plain/metrics.jsonl ++logging.events_path=runs/smoke/plain/events.jsonl
#     ... Training complete. Final loss: 4.962428569793701   (tracked signal moments, 2026-09-05; was 5.054330348968506 before)

# 3b. replay on — expect replays in the recurrent phases, none in novel_inject
python train.py --config-name stream_smoke ++controller.retrieval.replay_on_hit=true \
    ++run.dir=runs/smoke/replay ++checkpoint.dir=runs/smoke/replay/ckpt \
    ++logging.path=runs/smoke/replay/metrics.jsonl ++logging.events_path=runs/smoke/replay/events.jsonl
#     reference: 5 rows replayed, 20 P decisions, final loss 4.934509754180908 (2026-09-05, tracked moments; fewer R decisions than before so fewer replays)

# 3c. bit-exact resume with replay on — final loss identical to 3b to the last digit
python train.py --config-name stream_smoke ++controller.retrieval.replay_on_hit=true \
    ++checkpoint.resume_from=50 \
    ++run.dir=runs/smoke/resume ++checkpoint.dir=runs/smoke/replay/ckpt \
    ++logging.path=runs/smoke/resume/metrics.jsonl ++logging.events_path=runs/smoke/resume/events.jsonl

# 3d. corrupted-teacher stream — decisions attributed to the tag, digest changes
python train.py --config-name stream_smoke ++data.corrupt_teacher.enabled=true \
    ++data.corrupt_teacher.tag=math_teacher_corrupted ++controller.retrieval.replay_on_hit=true \
    ++run.dir=runs/smoke/corrupt ++checkpoint.dir=runs/smoke/corrupt/ckpt \
    ++logging.path=runs/smoke/corrupt/metrics.jsonl ++logging.events_path=runs/smoke/corrupt/events.jsonl

# 3e. config gate for the real stream (metadata only, ~1 min, no download)
python -m hivemind preflight --config-name stream_small --metadata-only
for m in baselines_small r_tier_small budget_sweep_small p_study_small ablation_grid bad_teacher_small; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --dry-run    # runs preflight, prints the matrix
done
```

Inspect any smoke run (replace the directory):

```bash
python - <<'PY'
import json, collections
d = "runs/smoke/replay"
ev = [json.loads(l) for l in open(f"{d}/events.jsonl")]
print(collections.Counter(e["action"] for e in ev if e["type"] == "decision"))
print("replays:", sum(e["type"] == "replay" for e in ev),
      "merges:", sum(e["type"] == "consolidation" for e in ev),
      "teachers:", collections.Counter(e["teacher"] for e in ev if e["type"] == "decision"))
print(json.load(open(f"{d}/run_summary.json"))["extra"])
PY
```

Expected: 3a prints the reference loss to the last digit, with P
decisions and `consolidations` ≥ 1 (the signals are alive); 3b shows
`replays: 20` (5 rows × 4 modules) and `replayed_total: 5`, with no
replay at steps 40–44 (`novel_inject`); 3c prints the same final loss as
3b (`4.934509754180908`); 3d shows `math_teacher_corrupted` among the teachers and a different
`stream_digest` from 3b.

## 3.5 Restart notice (2026-09-05)

Two defects were found by the first real runs and fixed the same day;
both are in `docs/CHANGELOG_iclr.md` (2026-09-05 entries):

- **The controller was reading dead Adam state.** Under exact masking a
  never-opened base-weight module has m = v = 0 forever, so surprise was
  pinned at `s_max` and C̄ = 0 at every decision, and P could never fire.
  The controller now tracks its own bias-corrected Adam-style moments
  (`controller.moments`, default `tracked`; ≈ 0.6 GB extra per run and per
  checkpoint at `sketch_stride: 8`). **Every E1 run made before this
  commit routed on repetition alone: delete `runs/baselines_small_v1` and
  restart from step 4.**
- **Live KD was zeroed** by the cache-hit mask (`f9b422d`): delete
  `runs/live_kd_small_v1` and rerun Tier 4.

Sanity check after `git pull`, on either VM, before relaunching:

```bash
python -m pytest tests/test_moments.py tests/test_training_loop_hf.py -q   # both green
python scripts/diagnose_p.py <any finished run>    # old runs: C̄ max 0.000 everywhere
```

On the new runs, `diagnose_p.py` must show C̄ spread across (0, 1) from
`general_warm` on, surprise well below 20 for most decisions, and a
non-zero P count by the end of `code_recurrent`.

## 4. The first proper training run

The headline controller: `trihope` — Qwen3-0.6B on `stream_small`
(6000 steps, 7 phases), full R/F/P from Adam state, replay on, sustained
stability so P fires. Everything else depends on it (its action shares
feed `random_routing`; its peak memory sets `--concurrent N`).

```bash
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --only trihope
```

Outputs, per seed: `runs/baselines_small_v1/trihope-seed<seed>/{metrics.jsonl,
events.jsonl, run_summary.json, checkpoints/, stdout.log, status.json}`.
Runtime ≈ 1.25–1.6 h per seed on an A100/L40S-class card.

The same run by hand, one seed, if you want to watch it:

```bash
python train.py --config-name stream_small controller.retrieval.replay_on_hit=true \
    ++train.seed=1337 ++run.dir=runs/first ++checkpoint.dir=runs/first/checkpoints \
    ++logging.path=runs/first/metrics.jsonl ++logging.events_path=runs/first/events.jsonl
```

### 4.1 Health signals (phase map: general_warm 0–499 · code_recurrent 500–1999 · novel_inject 2000–2149 · math_recurrent 2150–3649 · medical_recurrent 3650–4949 · code_revisit 4950–5399 · mixed_tail 5400–5999)

| When | Healthy | Wrong |
|---|---|---|
| step 0 | `router/miss_rate == 0` in metrics | any miss: `teachers.teacher_ids` do not match the data |
| general_warm | almost all decisions `R`; loss falls from ~10 | `F` everywhere from step 0 |
| code_recurrent | `F` takes over as repetition rises; `write/replay_count > 0` after ~step 600; first `P` decisions and `consolidation` events at the 250-step sweeps | `replayed_total` still 0 at step 1000; `consolidations == 0` at step 2000 |
| novel_inject | back to ~100 % `R`; **zero** replays (unique buckets by design) | replays inside this phase |
| math / medical | `eval/code/loss` rises a little, does not collapse; medical is where the confidence gate fires | code retention delta growing without bound |
| code_revisit | `eval/code/loss` recovers within a few hundred steps (50-step eval interval here) | never recovers |
| end | `run_summary.json` has `permanent_writes.total > 0`, `extra.retrieval.replayed_total > 0`, `retention.history` with 7 entries, `ledger_totals.teachers` | missing summary = the run did not finish (`status.json` says `failed`) |

Live view while it runs:

```bash
tail -f runs/baselines_small_v1/trihope-seed1337/stdout.log
python - <<'PY'
import json
rows = [json.loads(l) for l in open("runs/baselines_small_v1/trihope-seed1337/metrics.jsonl")]
last = [r for r in rows if "loss/total" in r][-1]
print({k: last[k] for k in ("step", "phase", "loss/total", "write/r_count", "write/f_count",
                            "write/p_count", "write/replay_count") if k in last})
PY
```

### 4.2 Right after it finishes

```bash
python -m analysis.run_report runs/baselines_small_v1        # shares file for random_routing + first tables
python - <<'PY'
import json; s = json.load(open("runs/baselines_small_v1/trihope-seed1337/run_summary.json"))
print("peak GB:", s["profile"].get("final_peak_mem_gb"), "wall h:", round(s["profile"]["wall_clock_s"]/3600, 2))
PY
```

Pick `N = floor(80 / peak GB)` for `--concurrent N`. Observed 2026-09-05
on a 96 GB card: ~12 GB for `trihope`, so N = 5 is safe (`full_ft` peaks
higher; N processes also share compute, so each runs ~N× slower than the
4 it/s a lone run shows).

To fill the card while the trihope seeds are still running, launch the
rest of E1 now and leave `random_routing` for last — it reads trihope's
action-share file and silently falls back to uniform thirds if the file
is missing (a different, weaker control than the paper claims):

```bash
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --resume --skip random_routing --concurrent 5
```

When the three trihope seeds are done, build the shares file and run the
last spec:

```bash
python -m analysis.run_report runs/baselines_small_v1
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --resume --only random_routing --concurrent 3
```

## 5. The full campaign, in order

```bash
# Tier 1 — E1 controller comparison (33 runs incl. the two above; go/no-go)
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --resume --concurrent 3
python -m analysis.run_report runs/baselines_small_v1

# Tier 2 — E5 bad teacher + rollback (21 runs + rollbacks)
python scripts/run_experiment.py configs/experiments/bad_teacher_small.yaml --concurrent 3
for seed in 1337 2024 7; do
  r=runs/bad_teacher_small_v1/trihope-seed$seed
  python scripts/rollback_teacher.py --run $r --teacher math_teacher_corrupted --out $r-rollback
  python scripts/rollback_teacher.py --run $r --teacher math_teacher_corrupted --baseline full_restore --out $r-fullrestore
  g=runs/bad_teacher_small_v1/gradient_routing-seed$seed
  python scripts/rollback_teacher.py --run $g --teacher math_teacher_corrupted --baseline full_restore --out $g-fullrestore
done
python -m analysis.run_report runs/bad_teacher_small_v1

# Tier 3 — E2 / E3 / E4 + signal ablations (15 + 22 + 5 + 11 runs)
python scripts/run_experiment.py configs/experiments/r_tier_small.yaml --concurrent 3
python scripts/run_experiment.py configs/experiments/budget_sweep_small.yaml --concurrent 3
python scripts/run_experiment.py configs/experiments/p_study_small.yaml --concurrent 3
python scripts/run_experiment.py configs/experiments/ablation_grid.yaml --concurrent 3
python -m analysis.run_report runs/baselines_small_v1 runs/budget_sweep_small_v1 --out runs/figure1   # Figure 1
python -m analysis.run_report runs/r_tier_small_v1
python -m analysis.run_report runs/p_study_small_v1
```

`rollback_teacher.py` exits 0 with "nothing to roll back (containment
held)" when no merge is attributed ≥ `--min-share` (default 0.5) to the
corrupted teacher — that is a result, not an error; lower `--min-share`
(e.g. 0.3) to roll back a partially attributed merge. `gradient_routing`
never merges, so only `full_restore` applies to it; its "unlearning" is
the slice ablation already inside the run (the `ablation` event at step
3649).

## 6. What each experiment produces

| Experiment | Directory | Key artifacts |
|---|---|---|
| E1 controllers | `runs/baselines_small_v1/analysis/` | `budget_curve.csv`, `pareto_budget.png`, `forgetting_table.csv`, `action_share_by_phase.csv` |
| E5 bad teacher | `runs/bad_teacher_small_v1/analysis/` + `*-rollback/rollback_summary.json` | `containment.csv`, `containment_bars.png`, `teacher_attribution.csv` |
| E2 no-write tier | `runs/r_tier_small_v1/analysis/` | `budget_curve.csv` (steps_to_recover_code) |
| E3 budget curve | `runs/figure1/` | `pareto_budget.png` over E1 + sweep |
| E4 consolidation timing | `runs/p_study_small_v1/analysis/` | `p_selection_stats.csv`, `damage_recovery.csv`, `*_p_timeline.png` |

After every `run_report`, write `runs/<experiment>/RESULTS.md` (table,
figure paths, three sentences: what it shows, what it does not, anything
surprising) and fill the slot in `docs/STATUS.md`.

## 7. Runner controls and recovery

```bash
python scripts/run_experiment.py MANIFEST --dry-run              # matrix + preflight, launches nothing
python scripts/run_experiment.py MANIFEST --only a,b             # only these specs (all seeds)
python scripts/run_experiment.py MANIFEST --skip random_routing  # everything except these specs
python scripts/run_experiment.py MANIFEST --resume               # skip done, resume failed/interrupted bit-exactly
python scripts/run_experiment.py MANIFEST --concurrent N         # N runs sharing the one GPU
python scripts/run_experiment.py MANIFEST --max-hours H          # stop launching past the budget
```

- One Ctrl+C checkpoints the current run; `--resume` continues it with
  identical losses.
- Each run is its own subprocess; an OOM kills one run, not the matrix.
  If it OOMs, lower `--concurrent`.
- E5 keeps every tagged checkpoint (`checkpoint.keep_tagged=0`, ~1.3 GB
  each); prune after the rollbacks.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Key 'X' is not in struct` | the override key is not in the base config: use `++key=value` |
| `Stream config changed since the checkpoint was written (digest …)` | you changed data/stream/corruption config and asked to resume — a fresh run, not a resume |
| `router/miss_rate > 0` at step 0 | `teachers.teacher_ids` ≠ the dataset's `teacher_id` values (see `docs/experiments.md`) |
| `random_routing` warns "shares file not found" | run `--only trihope` + `run_report` first |
| `checkpoint … does not exist` from `rollback_teacher.py` | the run lacked `checkpoint_before_merge` / `keep_tagged=0` / `save_before_phases` — the E5 manifest sets all three |
| `teacher_partition needs lora.rank >= number of teachers` | rank must be ≥ 5 on the corrupted stream (`stream_small` ships 16) |
| replicas "diverged across ranks" (DDP only) | a controller input escaped the sync; file a bug — never lower `assert_rank_consistency` |
| `fatal error: Python.h: No such file or directory` on the first forward (Triton compiling its CUDA driver helper) | the Python dev headers are missing: `sudo apt-get install -y python3.X-dev build-essential` (X = your interpreter's minor version), then relaunch with `--resume`; the compile happens once and is cached |
| the runner prints the launch command and then nothing | by design: the training subprocess writes to `<run_dir>/stdout.log`, not the terminal — `tail -f` it; `status.json` says `running` |
| the progress bar sits at step 499 / 1999 / 2149 / … for minutes | phase-boundary eval: validation loss on all four domains plus exact-match generation (64 samples × 64 new tokens, token by token) on code, medical and math; the same pause recurs every 250 steps and every 50 steps inside `code_revisit` |
| `permanent_writes.total == 0` and no `consolidation` events at the end | P never fired: run `python scripts/diagnose_p.py <run_dir>` — it prints per phase how far R, C̄ and V sit from the policy and sweep thresholds, which tells you which conjunct to relax (`controller.policy.stability_high_C` / `stability_low_V` / `repetition_medium`, or the sweep's `min_stability_C` / `min_repetition`) |
| `F=6, R=0` throughout `general_warm` | benign for the pretrained student: general-text loss is already ~1.2, so surprise never reaches `surprise_high=2.0` and everything is a tentative F write; the decisive checks are `novel_inject` (steps 2000–2149) routing to R with zero replays and `code_recurrent` showing replays after ~step 600 |

## 9. Tier 4 — live logit-KD robustness check (appendix)

Every run above trains on cached teacher **text**: no logit files exist,
so the KD term is inert (`loss/kd` = 0.0 in metrics.jsonl) and the
objective is confidence-weighted CE on `teacher_output_text` — sequence-
level distillation. Tier 4 re-runs the headline controller and its E1
rivals with `teachers.mode=live`, where the four Qwen-vocabulary teachers
declared in `stream_small.yaml` score the same cached text and the student
matches their full distributions (τ² · KL at τ = 4, CE at half weight):

```bash
python scripts/run_experiment.py configs/experiments/live_kd_small.yaml --concurrent 2
python -m analysis.run_report runs/live_kd_small_v1
```

Five runs at one seed, ~2× the cache-mode cost each (teachers add ~18 GB
in bf16 and four extra forwards per step; no bitsandbytes needed — set
`teachers.pretrained.general.quantization=nf4` to save ~6 GB if it is
installed). Run after Tier 1; it does not affect the go/no-go. Health
signal at step 0: `loss/kd` > 0 and `loss/kd_row_hit_frac` = 1.0 — a live
run that shows 0.0 for both is training on text (that was a real bug,
fixed 2026-09-05; any live run made on an older commit must be deleted
and rerun). Compare
`action_share_by_phase.csv` and `forgetting_table.csv` with the cache-mode
rows — the appendix claim is that routing behaviour is not an artefact of
text-only supervision. Provenance of the cached text, what confidence
means per domain, and why this tier does not join Figure 1:
`docs/data_provenance.md`.

## 10. Optional last tier — 1.7B headline

Only if the GPU is idle after Tier 4. `configs/experiments/headline.yaml`
holds `trihope` and `trihope_no_hash` at one seed on `stream_headline`
(~6 h each); uncomment the placeholder and paste in the override list of
whichever E1 baseline sat closest to trihope on Figure 1.
`stream_headline.yaml` now carries the disabled `corrupt_teacher` block,
so an E5 manifest can target it as well.

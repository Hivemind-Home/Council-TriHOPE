# Running TriHOPE — every command, every config

The complete operational reference. `docs/experiments.md` explains *what*
each experiment answers; this file is *how to run it*.

**Read this first — the two rules that will bite you:**

1. **Run `hivemind preflight` before spending GPU time.** It reports every
   config/data mismatch at once. Skipping it is how a 6-hour Group D run
   dies at hour 0 on a bucket-cardinality error.
2. **`data.batch_size` is the GLOBAL batch.** Under DDP it is split across
   ranks and must divide by the rank count. Changing it changes the stream
   digest, so a checkpoint written at one batch size will not resume at
   another.

---

## 0. Setup

```bash
pip install -e ".[dev]"                       # core + test tooling
pip install -e ".[data,analysis]"             # HF datasets + report generation
pip install -e ".[logging]"                   # optional: wandb
pip install -e ".[unsloth]"                   # optional: the unsloth backbone

python -m pytest tests/ -q                    # 409 tests, CPU-only, ~1 min
python -m hivemind doctor                     # torch / CUDA / device count
```

No HF token is needed — all four dataset repos are public. Set `HF_TOKEN`
only to raise rate limits.

---

## 1. Preflight — always first

```bash
python -m hivemind preflight --config-name stream_small --metadata-only   # ~60 s, no download
python -m hivemind preflight --config-name stream_small                   # + data checks
```

| Mode | Checks | Cost |
|---|---|---|
| `--metadata-only` | repos resolve, phase domains declared, mixed weights, live-teacher coverage, `train.steps` vs phase sum | seconds |
| full | + bucket cardinality per recurrent phase, `teacher_ids` vs the real rows, exact-match probe counts, novel-row budget | one split download |

Exit code 1 on any blocking problem. `scripts/run_experiment.py` runs the
metadata-only pass once per manifest automatically (`--no-preflight` opts out).

---

## 2. Smoke tests — CPU, no GPU required

```bash
# Fastest possible sanity check: synthetic data, 5 steps, ~10 s
python train.py --config-name pilot_smoke

# The real one: REAL HF Layer-C data (math + medical), 70 steps, ~5 min
python train.py --config-name stream_smoke

# Bit-exact resume check — must end at the SAME final loss
python train.py --config-name stream_smoke checkpoint.resume_from=50
```

Reference run (2026-08-07, CPU): loss `10.95 → 5.05`, routing `R=4` through
step 59 then `F=4`, final loss `5.054330348968506` — and the resumed run
lands on the same value to the last digit. If yours doesn't, stop and
investigate before using any GPU.

---

## 3. Single-GPU training

```bash
python train.py --config-name stream_small       # Qwen3-0.6B, 6000 steps, ~1-1.5 h on A100/L40S
python train.py --config-name stream_headline    # Qwen3-1.7B, 12000 steps, ~6 h
```

Any config key can be overridden on the command line (Hydra syntax; `++`
adds a key the config does not declare):

```bash
python train.py --config-name stream_small \
    train.seed=2024 \
    data.max_rows_per_domain=10000 \
    controller.policy.top_m_modules=4 \
    ++checkpoint.resume_from=latest
```

### Config map

| Config | Student | Data | Steps | Where it runs |
|---|---|---|---|---|
| `pilot_smoke` | tiny in-repo | synthetic | 5 | CPU |
| `pilot` | 512d / 12L in-repo | synthetic | — | CPU/GPU |
| `pilot_hf_smoke` | tiny in-repo | real HF, 10 rows | 5 | CPU |
| `pilot_hf` | HF backbone | real HF | — | GPU |
| `pilot_hf_gpu` | Qwen3-1.7B | real HF | — | GPU, live 30B teachers |
| `pilot_hf_live` | HF backbone | real HF | — | GPU, live teachers |
| `pilot_unsloth` | Unsloth backbone | real HF | — | GPU (24 GB) |
| **`stream_smoke`** | tiny in-repo | real HF (math+medical) | 70 | **CPU, ~5 min** |
| **`stream_small`** | **Qwen3-0.6B** | real HF, 4 domains | **6000** | **1 GPU, ~1-1.5 h** |
| **`stream_headline`** | **Qwen3-1.7B** | real HF, 4 domains | **12000** | **1 GPU, ~6 h** |

The bold three are the paper pipeline. The `pilot_*` configs predate it and
are kept for component-level debugging.

---

## 4. Multi-GPU

**DDP only.** `deepspeed` and `fsdp` are rejected at startup with the
reason: they void Theorem 1 silently (DeepSpeed replaces the optimizer step
so the R/F/P write masks are never consumed; FSDP removes the parameter
objects the controller keys on). See `src/hivemind/distributed.py`.

### 4a. Many independent runs → one spec per GPU

**This is what you want for Groups A and B.** No collectives, no
batch-size constraint, no stream-digest change, and N× the *runs*:

```bash
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --parallel-gpus 4
python scripts/run_experiment.py configs/experiments/ablation_grid.yaml   --parallel-gpus 4
```

Respects an inherited `CUDA_VISIBLE_DEVICES` (Slurm/Lightning allocations),
caps itself at the visible device count, and refills a GPU slot as each run
finishes.

### 4b. One long run → DDP-shard it

For the headline runs, where there is nothing to parallelize across:

```bash
# Directly
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    train.py --config-name stream_small ++distributed.enabled=true

# Through the matrix runner
python scripts/run_experiment.py configs/experiments/headline.yaml --nproc-per-node 4
```

`data.batch_size` must divide by the rank count. `stream_headline` ships
`batch_size: 1` (tuned for one GPU), so the runner **refuses to launch**
and prints the override to add:

```yaml
# configs/experiments/headline.yaml
  - id: trihope
    overrides: [data.batch_size=4]        # keeps per-device batch at 1 on 4 ranks
```

That changes the stream digest, so it is a **fresh run, not a resume**.

### 4c. The `distributed:` block

```yaml
distributed:
  enabled: false            # master switch; every collective no-ops at world_size 1
  backend: nccl             # gloo for the CPU equivalence tests
  strategy: ddp             # deepspeed/fsdp rejected at startup, even when disabled
  world_size: auto          # cross-checked against the launcher's WORLD_SIZE
  assert_rank_consistency: 100   # 0=off. LEAVE THIS ON — see below
  eval_mode: rank0
  consolidation_mode: replicate  # | rank0_broadcast
  on_empty_batch: fail           # | skip
  allow_world_size_change: false
  timeout_minutes: 30
  find_unused_parameters: false
  gradient_as_bucket_view: true
  broadcast_buffers: false
```

**Leave `assert_rank_consistency` on.** Replicas drifting apart produces no
error of its own — the run finishes looking healthy with a silently wrong
model. Tier 1 compares the per-step routing decisions on every step
(~40 µs); tier 2 compares an index-weighted checksum of params, `exp_avg`
and `coord_step` every N steps.

Measured 1-rank vs 2-rank on gloo: routing decisions and `coord_step`
identical at every horizon; the float gap starts at one ulp (3.7e-9 after
one step) and reaches ~4e-5 after 40, which is Adam amplifying
reduction-order noise. *Within* one job all ranks stay bit-identical.

`train.device` must be `auto` or bare `cuda` under a launcher — an explicit
`cuda:0` would pin every rank to device 0 and is rejected.

---

## 5. The experiment matrix

Run in this order. Each writes
`runs/{experiment}/{spec}-seed{seed}/{metrics.jsonl, events.jsonl,
run_summary.json, checkpoints/, stdout.log, status.json}`.

```bash
# A — baselines (paper Table 1 analogue): trihope, full_ft, lora_only,
#     no_retrieval, no_consolidation, gold_ce
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --parallel-gpus 4

# C — P-store study: p_study, p_off_control, p_forced_bad,
#     p_forced_plausible, p_forced_lowconf
python scripts/run_experiment.py configs/experiments/p_study_small.yaml --parallel-gpus 4

# B — signal ablations (needs A's `trihope` as the baseline)
python scripts/run_experiment.py configs/experiments/ablation_grid.yaml --parallel-gpus 4

# Tier 4 — live logit-KD robustness check (appendix; see docs/data_provenance.md)
python scripts/run_experiment.py configs/experiments/live_kd_small.yaml --concurrent 2

# D — headline (Qwen3-1.7B), optional
python scripts/run_experiment.py configs/experiments/headline.yaml
```

### Runner flags

```bash
--dry-run                 # print the matrix + status, launch nothing
--resume                  # skip done runs, resume failed ones from the latest checkpoint
--only trihope            # a single spec id
--max-hours 8             # stop launching past this budget (honoured in both modes)
--no-preflight            # skip the config gate
--parallel-gpus 4         # one spec per GPU (matrix)
--nproc-per-node 4        # DDP-shard each run (long runs)
```

Every run is its own subprocess, so an OOM kills one run rather than the
matrix. A single Ctrl+C checkpoints the current run first.

Seeds: manifests ship `seeds: [1337]` for a one-seed validation pass.
Change to `seeds: [1337, 2024, 7]` for the mean±std tables.

---

## 6. Resume

```bash
python train.py --config-name stream_small ++checkpoint.resume_from=latest
python train.py --config-name stream_small ++checkpoint.resume_from=3000        # a step
python train.py --config-name stream_small \
    ++checkpoint.resume_from=runs/.../checkpoints/step_00002075_pre_merge       # a path
python scripts/run_experiment.py MANIFEST --resume                               # whole matrix
```

Resume is bit-exact: the resumed run ends at the same loss as an
uninterrupted one (enforced by `tests/test_resume_exact.py`). It refuses
when the stream digest changed (different phases, repos, `bucket_columns`
or row cap), or when the world size changed
(`distributed.allow_world_size_change=true` overrides — the batch
composition survives but the float reduction order does not, so it is no
longer bit-exact).

`checkpoint.checkpoint_before_merge=true` writes a tagged
`step_XXXXXXXX_pre_merge` directory before every consolidation. Resume from
it to run the counterfactual where the promotion never happened.

---

## 7. Analysis

```bash
python -m analysis.run_report runs/baselines_small_v1
# → runs/baselines_small_v1/analysis/{report.md, *.csv, *.png}
```

| Output | Paper artifact |
|---|---|
| `forgetting_table.csv` | Table 1 analogue |
| `action_share_by_phase.csv` | Table 2 analogue (R/F/P shares per phase) |
| `ablation_deltas.csv` | ablation study |
| `p_selection_stats.csv` | when/why P fires |
| `adapter_reuse_aulc.csv` | adapter reuse after consolidation |
| `damage_recovery.csv` | forced-merge damage + recovery |
| `*_actions.png` | Figure 3 analogue |

---

## 8. Knobs you will actually reach for

| Key | Meaning |
|---|---|
| `train.seed` | run seed |
| `train.trainable` | `all` \| `lora` (the LoRA-only baseline) |
| `train.mixed_precision.dtype` | `bf16` (fp16 is rejected under DDP) |
| `data.batch_size` | **global** batch |
| `data.max_rows_per_domain` | per-domain row cap (`null` = full) |
| `data.gold_policy` | `teacher_only` \| `gold_when_available` |
| `data.strict_load` | a missing domain fails instead of warning |
| `data.shuffle_before_cap` | sample the corpus, not the head of the parquet |
| `data.domains[].bucket_columns` | synthesize a finer bucket key (code needs this) |
| `distillation.ce_confidence_weighting` | scale CE by teacher confidence |
| `controller.enabled` | `false` = the full-FT / LoRA baselines |
| `controller.policy.top_m_modules` | how many blocks get an action per step |
| `controller.consolidation.period` | steps between consolidation sweeps |
| `controller.ablation.disable_stores` | `[R]`, `[P]`, … |
| `controller.debug.force_consolidate_steps` | force a merge at given steps |
| `optim.masked` | `false` disables write masking (MaskedAdamW → plain AdamW) |
| `eval.exact_match.max_scan_rows` | how many val rows to scan for gold probes |

---

## 9. Before the campaign

```bash
python -m pytest tests/ -q                                        # 409 tests
python -m hivemind preflight --config-name stream_small           # must print clean
python -m hivemind preflight --config-name stream_headline
python train.py --config-name stream_smoke                        # ~5 min
python train.py --config-name stream_smoke checkpoint.resume_from=50   # same final loss

rm -rf checkpoints/ runs/    # the digest now covers data identity — old
                             # checkpoints cannot resume and would only confuse
```

Then GPU, in order: single-GPU `stream_smoke` → 2-GPU `stream_smoke` with
`++distributed.assert_rank_consistency=1` → `--parallel-gpus N` on
`baselines_small.yaml` (`--dry-run` first) → the headline.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `needs 8 buckets with ≥16 rows … found 3` | the corpus has fewer buckets than the phase requests | add `bucket_columns` to that domain, or lower `num_buckets` |
| `data.batch_size=1 … does not divide by --nproc-per-node 4` | global batch not shardable | add `overrides: [data.batch_size=4]`; it is a fresh run |
| `Stream config changed since the checkpoint was written` | phases, repos, `bucket_columns` or the row cap changed | start fresh, or revert the config |
| `checkpoint … does not match the student model` | model config changed, or a DDP-wrapped module was passed | fix the config; `checkpoint.allow_partial_load=true` only if you mean it |
| `requires the unwrapped student module` | a DDP wrapper reached save/load or `build_optimizer` | pass `ddp.module` |
| `routing decisions diverged across ranks` | replicas disagree — a real correctness failure | do not ignore; open an issue with the step and payload |
| `distributed.enabled=true but … not started by a launcher` | ran `python train.py` instead of `torch.distributed.run` | use the launcher, or set `enabled: false` |
| `exact_match: domain 'x' yielded N probes` | gold rows are sparse in the scan window | raise `eval.exact_match.max_scan_rows` |
| `MetadataRouter missed N/M samples` | `teachers.teacher_ids` ≠ the dataset's values | copy the real ids from `docs/experiments.md` |
| `tokenizer.vocab_size != model.vocab_size` (warning) | expected for Qwen3 (151643 vs 151936) | ignore; the collator uses the model vocab |
| `strategy='fsdp' is not supported` | FSDP/DeepSpeed void Theorem 1 | use `ddp` |

## ICLR-2027 additions (branch `iclr2027-push`)

New knobs, all defaulting to the pre-existing behaviour; full details per
task in `docs/CHANGELOG_iclr.md`, campaign order in `docs/GPU_RUNBOOK.md`.

| Key | What it does |
|---|---|
| `controller.policy.stability_source: instant\|sustained` | P compares the sustained cosine C̄ (EMA) instead of the per-step cosine |
| `controller.policy.mode: rfp\|surprise_only\|adam_score\|teacher_partition` | controller family (TriHOPE / Titans-style / MoLF-style / Gradient-Routing-style) |
| `controller.policy.adam_score_high` | P threshold on mean m²/v for `adam_score` |
| `controller.retrieval.replay_on_hit` (+ `hit_threshold`, `replay_batches`) | R → F replay: parked rows are replayed into LoRA inside the step once their bucket recurs |
| `controller.writer.flag_p_for_consolidation` | `false` = a P action on an adapter never flags a merge (MoLF-style) |
| `controller.consolidation.trigger: signals\|plateau` (+ `plateau_*`) | loss-plateau merge of all adapters (Online-LoRA-style); works with the controller off |
| `controller.debug.policy_override: random_matched` (+ `random_shares_path`, `random_unit`) | budget-matched random routing from a run's `action_share_by_phase.csv` |
| `controller.debug.block_p_for_teachers` (+ `block_min_share`) | selective rollback: demote the teacher's P, reset its dominated adapters on resume |
| `controller.debug.ablate_teacher_slice: {step, teacher}` | gradient-routing ablation of one teacher's rank slice |
| `data.corrupt_teacher.{enabled, domain, phase, fraction, mode, confidence, tag}` | corrupted-teacher stream (E5); changes the digest when enabled |
| `checkpoint.save_before_phases: [phase]` | tagged `pre_phase_<name>` checkpoint before a phase |
| `checkpoint.keep_tagged: 0` | keep every tagged (`pre_merge`, `pre_phase_*`) checkpoint |
| `train.skip_step_ranges: [[lo, hi]]` | consume but do not train on those steps (full-restore counterfactual) |
| `eval.interval_by_phase: {phase: steps}` | per-phase eval interval (50 inside `code_revisit`) |
| `scripts/run_experiment.py --concurrent N` | N runs sharing the one visible GPU |
| `scripts/rollback_teacher.py` | selective rollback / full restore from a finished run |
| `python -m analysis.run_report DIR [DIR ...]` | pooled report; `budget_curve.csv`, `pareto_budget.png`, `containment*.{csv,png}` |


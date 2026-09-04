# TriHOPE Experiments — Reproducibility Guide

This document describes every experiment behind the paper's empirical
claims: what question it answers, the exact command, and the artifacts it
produces.

> **Looking for commands?** [docs/running.md](running.md) is the full
> operational reference — setup, preflight, smoke tests, single- and
> multi-GPU, resume, the config map, every knob, and troubleshooting. All experiments are driven by manifest files in
`configs/experiments/` and executed by `scripts/run_experiment.py`.

## Setup

```bash
pip install -e ".[data,analysis]"     # + [logging] for wandb
python -m pytest tests/ -q            # 499 tests, CPU-only, ~30s
```

Datasets (public, HuggingFace hub, downloaded automatically on first use):

| Domain  | Repo                                    | Splits | Buckets | Gold | Conf mean |
|---------|-----------------------------------------|--------|---------|------|-----------|
| general | `hivemind-research/general-layerC-200k` | 160k/20k/20k | 27 | 0% | 0.818 |
| code    | `hivemind-research/code-layerC-200k`    | 160k/20k/20k | **3** native / 12 composite | 100% | 0.829 |
| math    | `hivemind-research/math-layerC-200K`    | 160k/20k/20k | 15 | 2% | 0.875 |
| medical | `hivemind-research/medical-layerC-200k` | 160k/20k/20k | 17 | 78% | **0.596** |

All four share an identical 18-column schema. `bucket_id` is
`{domain}_{subdomain}_{difficulty}`.

**Two consequences worth knowing before reading any result.**

*Code needs a composite bucket key.* The code corpus assigns a single
subdomain to every row, so its published `bucket_id` has only 3 distinct
values — fewer than the 8 a recurrent phase requests. The stream configs
therefore set `bucket_columns: [subdomain, difficulty, source]` for code
only, which yields 12 buckets (all with ≥16 rows at the 40k cap).
`source` (codeforces / atcoder / codechef / …) is a genuine problem-style
cluster, exactly as `subdomain` is for math. Every other domain keys on
its published `bucket_id`.

*Medical carries the confidence signal.* It is the only domain whose
`teacher_confidence` reaches the 0.3 policy gate (mean 0.596 vs 0.82-0.88
elsewhere; 5.6% of rows below the threshold). Without it in a recurrent
phase, the `no_teacher_conf` ablation is a null result by construction
rather than by science.

Layer C = pre-joined canonical samples (Layer A) × teacher supervision
(Layer B): `input_text, target_text, has_gold_label, bucket_id,
teacher_id, teacher_output_text, teacher_confidence, ...` — see
`docs/dataset_preperation.md`.

**Teacher supervision modes.** All matrix runs use **cache mode**: the
student trains with CE on `teacher_output_text`, scaled per row by
`teacher_confidence` (`distillation.ce_confidence_weighting`). Note that
`use_teacher_confidence` alone reaches only the KD term, and KD is inert
without a logit cache — the NPZ files referenced by `teacher_logits_path`
are not published — so `ce_confidence_weighting` is the only route by
which confidence changes the objective in cache mode.

`teachers.teacher_ids` must equal the values in the tables:

| domain | teacher_id |
|---|---|
| general | `general_teacher_deepseek_r1_distill_llama_70b` |
| code | `code_teacher_deepseek_r1` |
| math | `math_teacher_deepseek_r1_distill_qwen_1p5b` |
| medical | `medical_teacher_qwen2p5_1p5b_instruct_clean` |

The one **live** run (`trihope_live_kd`) loads four small domain experts
for true temperature-scaled logit KD. `teacher_hf.py` hard-raises unless
`vocab_size == 151936`, which rules out DeepSeek-R1-Distill-Llama-70B
(128256) and the 7B Qwen2.5 Coder/Math variants (152064) — only the 1.5B
Qwen2.5 models qualify. The shipped set totals ~12 GB and fits alongside
the 1.7B student on a 40 GB card; math and medical are exactly the models
that produced the cached supervision, so live and cache stay comparable.

## Preflight (run this before ANY GPU time)

```bash
hivemind preflight --config-name stream_small --metadata-only   # ~60 s, no download
hivemind preflight --config-name stream_small                   # + data checks
```

Validates the config against the real data and reports **every** problem at
once: bucket cardinality against what each recurrent phase asks for,
`teacher_ids` against the values actually in the rows, exact-match probe
availability, the novel-row budget, undeclared phase domains, live-teacher
coverage. `scripts/run_experiment.py` runs the metadata-only pass once per
manifest before launching anything, so a broken config fails in a minute
instead of six hours into Group D.

## Warm-up smoke (run this second)

```bash
python train.py --config-name stream_smoke        # CPU, ~5 min
python train.py --config-name stream_smoke checkpoint.resume_from=50   # resume check
```

Trains a tiny random-init student on real math + medical Layer-C data
through a 5-phase stream (warm → recurrent → novel → cross-domain
recurrent → revisit). Verifies: Layer-C loading, stream scheduling, R/F/P
routing + event trace, phase-boundary eval, checkpoint/resume (the resumed
run must end at the **exact** same loss), and `logs/run_summary.json`.

Expected routing behavior: ~100% R in the warm/novel phases, F once
repetition rises in the recurrent phases. A reference run (2026-08-07,
CPU, ~5 min): loss 10.95 → 5.05, `R=4` through step 59, `F=4` from step
60, and the resumed run finished at `5.054330348968506` — identical to the
uninterrupted run to the last digit.

Code is deliberately absent from the smoke: `load_dataset` fetches the
whole split before `.select`, and `code-layerC-200k` is 64 shards
(~870 MB). Its composite-bucket path is covered without any download by
`tests/test_stream_configs.py`.

## The stream

`stream_small` (6000 steps, Qwen3-0.6B) and `stream_headline` (12000
steps, Qwen3-1.7B) share the same phase structure:

| Phase | Mode | Domain | small / headline | Purpose |
|---|---|---|---|---|
| general_warm | random | general | 500 / 1000 | settle optimizer statistics |
| code_recurrent | recurrent | code | 1500 / 3000 | 8 fixed buckets revisited every 10 steps (+50% fresh rows on background slots) → repetition/stability rise → F and eventually P |
| novel_inject | novel | general | 150 / 300 | isolated single-use rows with unique bucket ids → should route to R |
| math_recurrent | recurrent | math | 1500 / 3000 | second recurrent domain (tests forgetting of code) |
| medical_recurrent | recurrent | medical | 1300 / 2600 | third recurrent domain; the only one that exercises the teacher-confidence gate |
| code_revisit | recurrent | code | 450 / 900 | *the same* buckets as code_recurrent (`reuse_from`), now after **two** intervening domains → steps-to-recover |
| mixed_tail | mixed | all 4 | 600 / 1200 | interleaved replay tail |

Every recurrent phase sets `exclude_chosen_buckets_from_background: true`.
Without it only the *sampled rows* are reserved, so a background draw can
still land on another row of a chosen bucket — and since the controller
keys repetition on the bucket, those hits inflate the very counter the
phase exists to measure. With `n/(n+k)` at `k=10` the frequency term
saturates above 0.9 after ~45 such hits.

Streams are deterministic given the seed; the schedule digest is stored in
every checkpoint and validated on resume. The digest covers the *data*
identity as well as the phase list (repos, `bucket_columns`, row cap,
split), so a run can no longer swap its corpus and still pass the guard.

## Experiment groups

> **ICLR-2027 campaign:** the tiered run order for the reframed paper
> (E1 controller comparison, E5 bad-teacher rollback, E2/E3/E4) lives in
> [docs/GPU_RUNBOOK.md](GPU_RUNBOOK.md). `baselines_small.yaml` now holds
> the four controller baselines (`surprise_gate`, `molf_style`,
> `plateau_trigger`, `random_routing`) plus `trihope_no_hash`;
> `r_tier_small.yaml` (E2), `budget_sweep_small.yaml` (E3) and
> `bad_teacher_small.yaml` (E5) are new. `python -m analysis.run_report`
> accepts several experiment dirs and writes `budget_curve.csv` +
> `pareto_budget.png` (Figure 1). Per-task details: `docs/CHANGELOG_iclr.md`.

Run in this order. Each run writes
`runs/{experiment}/{spec}-seed{seed}/{metrics.jsonl, events.jsonl,
run_summary.json, checkpoints/, stdout.log, status.json}`.

### Group A — Baselines (paper Table 1 analogue)

```bash
python scripts/run_experiment.py configs/experiments/baselines_small.yaml
```

| Spec | What it is |
|------|-----------|
| `trihope`          | full controller |
| `full_ft`          | controller off, no LoRA, every base weight trains (`controller.enabled=false model.lora.rank=0`) |
| `lora_only`        | controller off, base+shared frozen, all adapters always open (`train.trainable=lora`) |
| `no_retrieval`     | R branch disabled — R-routed batches fall through to F |
| `no_consolidation` | P branch + periodic sweep disabled |

### Group C — Permanent-memory (P-store) study

```bash
python scripts/run_experiment.py configs/experiments/p_study_small.yaml
```

Answers the paper's four P questions:

1. **When is P selected?** → `events.jsonl` `decision`/`consolidation`
   records; `analysis.tables.p_selection_stats` reports counts, first-fire
   steps, and the signal values at P moments. The `p_study` spec lowers
   the consolidation thresholds (period 250, min C/R 0.4) so P fires
   measurably within the 6000-step stream.
2. **Does P improve retention/efficiency?** → `p_study` vs
   `p_off_control` (identical routing thresholds, P disabled):
   phase-boundary retention deltas + steps-to-recover on `code_revisit`.
3. **Does consolidation permit adapter reuse?** →
   `analysis.tables.adapter_reuse_aulc` compares area-under-training-loss
   on `math_recurrent` (the phase after code consolidations freed the
   adapters) between `p_study` and `p_off_control`.
4. **Does incorrect consolidation cause irreversible damage?** →
   `p_forced_bad` force-merges ALL adapters mid-`novel_inject` (maximally
   wrong, step 2075), `p_forced_plausible` early in `math_recurrent`
   (wrong-but-plausible, step 2350), `p_forced_lowconf` 200 steps into
   `medical_recurrent` (step 3850 — the one regime where teacher
   confidence dips below the gate that was meant to prevent this).
   These are offsets into the phase map, so they move if phase lengths do. `eval.on_consolidation=true` measures damage at
   merge resolution; `analysis.tables.damage_recovery` reports the eval
   jump and steps-to-recover. Every merge is preceded by a tagged
   `step_XXXXXXXX_pre_merge` rollback checkpoint — resume from it
   (`checkpoint.resume_from=<path>`) to run the counterfactual in which
   the promotion never happened.

### Group B — Routing-signal ablations

```bash
python scripts/run_experiment.py configs/experiments/ablation_grid.yaml
```

One run per signal: `no_surprise`, `no_repetition`, `no_cosine`,
`no_volatility`, `no_teacher_conf`, plus `topm_2`/`topm_all` (Top-M module
selection) and `topk_25`/`topk_100` (Top-K rank selection). A disabled
signal is *not computed* and is pinned to a neutral value chosen so the
policy degrades to the sub-policy over the remaining signals (see
`AblationConfig` in `src/hivemind/controller/config.py` for the exact
semantics). `analysis.tables.ablation_deltas` reports each ablation's
deltas vs `trihope` — run Group A first (or add a `trihope` spec) so the
baseline exists.

Group A also carries `gold_ce`, which sets
`data.gold_policy=gold_when_available`: rows the dataset marks
`has_gold_label` are supervised on their own `target_text` instead of the
teacher trace. That flag is 100% true on code and 78% on medical, so it is
a genuinely different objective and is reported as an ablation rather than
being made the default. `docs/dataset_preperation.md`'s "gold → CE+KD, no
gold → KD only" design is aspirational; this run is where it is measured.

### Group D — Headline (Qwen3-1.7B)

```bash
python scripts/run_experiment.py configs/experiments/headline.yaml
```

Same five baselines at 1.7B/12000 steps, plus `trihope_live_kd` — the one
true logit-KD run with live nf4 30B teachers.

## Runner controls

```bash
python scripts/run_experiment.py MANIFEST --dry-run        # print matrix + status
python scripts/run_experiment.py MANIFEST --resume         # skip done, resume failed from latest checkpoint
python scripts/run_experiment.py MANIFEST --only trihope   # single spec
python scripts/run_experiment.py MANIFEST --max-hours 8    # session budget
python scripts/run_experiment.py MANIFEST --no-preflight   # skip the config gate
# equivalently: hivemind experiment MANIFEST [...]
```

## Multi-GPU

Two ways to spend N GPUs, with different trade-offs. **Pick by workload,
not by habit.**

```bash
# Groups A/B — many INDEPENDENT runs. One spec per GPU beats DDP outright:
# no collectives, no batch-size constraint, no digest change, N x the runs.
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --parallel-gpus 4

# Group D / live-KD — ONE long run. DDP-shard it. data.batch_size is the
# GLOBAL batch and must divide by the rank count, so the headline (which
# ships batch_size: 1, tuned for one GPU) needs the override below to keep
# the per-device batch at 1. The runner refuses to launch without it.
python scripts/run_experiment.py configs/experiments/headline.yaml \
    --nproc-per-node 4      # -> add `overrides: [data.batch_size=4]` to the manifest

# Or directly:
python -m torch.distributed.run --standalone --nproc_per_node=4 \
    train.py --config-name stream_small ++distributed.enabled=true
```

**DDP is the only supported strategy, and `deepspeed`/`fsdp` are rejected
at startup with the reason.** TriHOPE's Theorem 1 (exact state isolation)
requires that the optimizer step stays ours and that `m`/`v` stay
unsharded. DeepSpeed's `engine.step()` replaces `MaskedAdamW.step()`, so
the staged write masks are never consumed and R/F/P authorization becomes
a *silent* no-op. FSDP with `use_orig_params=False` removes the parameter
objects `register_controller_params` keys on, flipping closed coordinates
to open with no error. DDP replicates the model, all-reduces gradients,
and shares the identical `nn.Parameter` objects with the wrapper, so the
module index, optimizer state and masks all keep working. See
`src/hivemind/distributed.py`.

`data.batch_size` is the **global** batch and must divide by the rank
count; the stream schedule is built at that global size on every rank and
sharded contiguously per step, so bucket purity, novel-row uniqueness and
the digest are all preserved. Changing `batch_size` changes the digest, so
a multi-GPU headline run (`batch_size: 1 → 4`) is a *fresh* run, not a
resume.

The controller's three rank-local inputs (`bucket_id`, `embedding`,
`conf_mean`) are made global by one fused all-reduce per step;
`distributed.assert_rank_consistency` (default 100) then verifies every
rank still agrees on the routing decisions. **Leave it on** — replicas
drifting apart produces no error of its own. Measured 1-rank vs 2-rank on
gloo: routing decisions and `coord_step` identical at every horizon; the
float gap starts at one ulp (3.7e-9 after one step) and reaches ~4e-5
after 40, which is Adam amplifying reduction-order noise, not a defect.

Every run is its own subprocess (an OOM kills one run, not the matrix).
Interrupting a run with a single Ctrl+C checkpoints it first; `--resume`
continues bit-exactly (same losses as an uninterrupted run — enforced by
`tests/test_resume_exact.py`).

Seeds: manifests ship with `seeds: [1337]` for the 1-seed validation pass;
change to `seeds: [1337, 2024, 7]` for the mean±std tables.

## Analysis

```bash
python -m analysis.run_report runs/baselines_small_v1
# → runs/baselines_small_v1/analysis/{report.md, *.csv, *.png}
```

| Output | Paper artifact |
|--------|----------------|
| `action_share_by_phase.csv` | Table 2 analogue (R/F/P shares per phase) |
| `forgetting_table.csv`      | Table 1 analogue (final loss/EM, worst retention delta, peak mem, wall-clock, write counts) |
| `ablation_deltas.csv`       | ablation study table |
| `p_selection_stats.csv`     | when/why P fires |
| `adapter_reuse_aulc.csv`    | adapter-reuse comparison |
| `damage_recovery.csv`       | forced-merge damage + recovery half-life |
| `*_actions.png`             | Figure 3 analogue (action composition over the stream) |
| `retention_curves.png`, `*_p_timeline.png`, `*_signals.png` | supporting figures |

## Audit trail

Every run leaves a complete record of *when and why parameters changed*:

- `events.jsonl` — one `decision` record per Top-M module per step (action,
  all signal values S/C/V/R + components, coords opened, bucket, teacher),
  `consolidation` records (strategy, pre-merge signals, forced flag),
  `phase_start/phase_end/phase_eval`, `resume` markers, `run_config`.
- `checkpoints/*/ledger.pt` — cumulative per-module counters
  (times opened, last opened step, total coords opened, consolidations).
- `run_summary.json` — wall-clock, tokens/sec, per-section time EMAs,
  controller-overhead fraction, per-phase peak CUDA memory, retention
  table, ledger totals.

## Formal-claim tests

The paper's theorems are enforced by the test suite:

| Claim | Test |
|-------|------|
| 1 rank ≡ N ranks (identical routing decisions + coordinate counters) | `tests/test_distributed_equivalence.py` |
| The rank-consistency guardrail can actually fire | `tests/test_distributed_equivalence.py::TestTheGuardrailIsNotVacuous` |
| Merges stay identical across ranks (both strategies) | `tests/test_distributed_consolidation.py` |
| Every shipped config builds against the real bucket shape | `tests/test_stream_configs.py` |
| A DDP-wrapped student never reaches save/load | `tests/test_checkpoint.py::TestWrappedModelRejected` |
| Theorem 1 (exact state isolation under MaskedAdamW) | `tests/test_masked_adamw.py::TestClosedCoordinateIsolation`, `tests/test_writer.py::test_unselected_module_invariant_through_masked_steps` |
| Corollary 1 (coordinate-local bias correction) | `tests/test_masked_adamw.py::TestCoordinateLocalBiasCorrection` |
| Lemma 1 (all-open reduction to AdamW) | `tests/test_masked_adamw.py::TestAllOpenReduction` |
| Theorem 2 (function-preserving consolidation) | `tests/test_consolidation.py::test_merge_is_function_preserving` |
| Corollary 2 (fresh optimizer state after merge) | `tests/test_consolidation.py::test_merge_resets_adapter_optimizer_state` |
| Proposition 1 (exact Top-K budget) | `tests/test_writer.py::test_f_action_opens_top_k_only` |
| Bit-exact resume | `tests/test_resume_exact.py` |

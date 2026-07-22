# TriHOPE Experiments — Reproducibility Guide

This document describes every experiment behind the paper's empirical
claims: what question it answers, the exact command, and the artifacts it
produces. All experiments are driven by manifest files in
`configs/experiments/` and executed by `scripts/run_experiment.py`.

## Setup

```bash
pip install -e ".[data,analysis]"     # + [logging] for wandb
python -m pytest tests/ -q            # 248 tests, CPU-only, ~10s
```

Datasets (public, HuggingFace hub, downloaded automatically on first use):

| Domain  | Repo                                    | Splits (train/val/test) | Notes |
|---------|-----------------------------------------|--------------------------|-------|
| general | `hivemind-research/general-layerC-200k` | 160k / 20k / 20k         | Layer C training-ready view |
| code    | `hivemind-research/code-layerC-200k`    | 160k / 20k / 20k         | HF viewer job failed; parquet loads fine |
| math    | `hivemind-research/math-layerC-200K`    | 160k / 20k / 20k         | |
| medical | *(not used)*                            | —                        | no 200k Layer C yet; `medical-layerC-final2` (32k) can be added as a 4th `DomainSpec` when wanted |

Layer C = pre-joined canonical samples (Layer A) × teacher supervision
(Layer B): `input_text, target_text, has_gold_label, bucket_id,
teacher_id, teacher_output_text, teacher_confidence, ...` — see
`docs/dataset_preperation.md`.

**Teacher supervision modes.** All matrix runs use **cache mode**: the
student trains with CE on `teacher_output_text`, weighted by
`teacher_confidence`; no teacher model is loaded. The one **live** run
(`trihope_live_kd` in the headline manifest) loads nf4-quantized 30B Qwen3
teachers for true temperature-scaled logit KD (needs a ≥40 GB GPU).

## Warm-up smoke (run this first)

```bash
python train.py --config-name stream_smoke        # CPU, ~4 min
python train.py --config-name stream_smoke checkpoint.resume_from=50   # resume check
```

Trains a tiny random-init student on real math Layer-C data through a
4-phase stream (warm → recurrent → novel → revisit). Verifies: Layer-C
loading, stream scheduling, R/F/P routing + event trace, phase-boundary
eval, checkpoint/resume (the resumed run must end at the **exact** same
loss), and `logs/run_summary.json`. Expected routing behavior: ~100% R in
the warm/novel phases, F once repetition rises in the recurrent phases.

## The stream

`stream_small` (6000 steps, Qwen3-0.6B) and `stream_headline` (12000
steps, Qwen3-1.7B) share the same phase structure:

| Phase           | Mode      | Domain  | Purpose |
|-----------------|-----------|---------|---------|
| general_warm    | random    | general | settle optimizer statistics |
| code_recurrent  | recurrent | code    | 8 fixed buckets revisited every 10 steps (+50% fresh rows on background slots) → repetition/stability rise → F and eventually P |
| novel_inject    | novel     | general | isolated single-use rows with unique bucket ids → should route to R |
| math_recurrent  | recurrent | math    | second recurrent domain (tests forgetting of code) |
| code_revisit    | recurrent | code    | *the same* buckets as code_recurrent (`reuse_from`) → steps-to-recover |
| mixed_tail      | mixed     | all     | interleaved replay tail |

Streams are deterministic given the seed; the schedule digest is stored in
every checkpoint and validated on resume.

## Experiment groups

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
   wrong), `p_forced_plausible` early in `math_recurrent`
   (wrong-but-plausible). `eval.on_consolidation=true` measures damage at
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
# equivalently: hivemind experiment MANIFEST [...]
```

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
| Theorem 1 (exact state isolation under MaskedAdamW) | `tests/test_masked_adamw.py::TestClosedCoordinateIsolation`, `tests/test_writer.py::test_unselected_module_invariant_through_masked_steps` |
| Corollary 1 (coordinate-local bias correction) | `tests/test_masked_adamw.py::TestCoordinateLocalBiasCorrection` |
| Lemma 1 (all-open reduction to AdamW) | `tests/test_masked_adamw.py::TestAllOpenReduction` |
| Theorem 2 (function-preserving consolidation) | `tests/test_consolidation.py::test_merge_is_function_preserving` |
| Corollary 2 (fresh optimizer state after merge) | `tests/test_consolidation.py::test_merge_resets_adapter_optimizer_state` |
| Proposition 1 (exact Top-K budget) | `tests/test_writer.py::test_f_action_opens_top_k_only` |
| Bit-exact resume | `tests/test_resume_exact.py` |

# Hivemind

**Multi-Teacher Knowledge Distillation with Tri-Store (R/F/P) Memory Routing.**

Hivemind explores a central research question: *can the internal states of adaptive optimizers serve as an interpretable window into LLM learning dynamics?* It formalizes three signal families — **surprise**, **stability**, and **repetition** — derived from Adam's moment estimates, and uses them to drive a knowledge-routing policy that decides, per module per step, *where* learned knowledge should be stored during multi-teacher distillation.

The codebase is two interlocking blocks:

- **Theory 201 — Distillation block.** Multi-teacher routing via cosine similarity (or dataset metadata), KD + CE + regularization loss, backpropagation. Produces the gradient the controller reads.
- **Theory 101 — Controller block.** Reads per-module gradients + Adam `m_{t-1}`, `v_{t-1}` (BEFORE the optimizer step) and emits `surprise / stability / repetition` per module. A top-M selection + R/F/P policy then routes the update to one of three stores.

Full theoretical write-ups live in [`docs/theory_201.md`](docs/theory_201.md) and [`docs/theory_101.md`](docs/theory_101.md). The dataset format is documented in [`docs/dataset_preperation.md`](docs/dataset_preperation.md).

## Core idea

During training, Adam maintains per-parameter exponential moving averages of gradients (`m`) and squared gradients (`v`). These are typically treated as optimization bookkeeping. Hivemind treats them as a free, interpretable per-parameter learning history:

| Signal | What it reads | What it means |
|---|---|---|
| **Surprise** `S_t(j)` | `(1/d_j) · Σ g²/v` | "Is this gradient unusually large for this module's history?" |
| **Stability** `C_t(j)` | `cos(g_t, m_{t-1})` | "Is this update aligned with prior direction or contradicting it?" |
| **Stability** `Stab_t(j)` | `(1/d_j) · Σ m²/v` | "How much of the gradient energy is consistent direction vs noise?" |
| **Stability** `V_t(j)` | EMA-windowed variance of ‖g‖ | "Is the gradient norm jumpy or smooth in time?" |
| **Repetition** `R_t(j)` | fused `R_mom + R_hash + R_ret` | "Has this pattern been showing up; is the model converging on it?" |

The fused per-module score drives a **tri-store routing policy**:

- **R-store (Retrieval)** — high surprise, low repetition: store sample embedding + teacher_id + bucket; no weight update.
- **F-store (Fast / LoRA)** — repeating but unstable: write into LoRA adapters, with optional Top-K masking on rank components or output rows.
- **P-store (Permanent)** — repeating and stable: consolidate F→P (direct merge or distillation-based; see "Consolidation").

## Pipeline

Each training step is 13 stages with one critical ordering constraint — the controller MUST read `m_{t-1}, v_{t-1}` AFTER `backward()` but BEFORE `optimizer.step()`:

```
Input batch (tokens, optional teacher_logits cache hits, optional metadata)
     │
     ▼
┌───────────────────────────────┐
│ 1. Shared routing embedding    │  f_emb(x) → R^d
└────────────┬──────────────────┘
             ▼
┌───────────────────────────────┐
│ 2. Teacher router              │  cosine prototypes  OR  metadata
│    (TeacherRouter | Metadata)  │       (cfg.data.router)
└────────────┬──────────────────┘
             ▼
┌───────────────────────────────┐
│ 3. Teacher forward             │  cache fast-path  OR  live forward
│    (CacheBacked | HF | toy)    │       (cfg.teachers.mode)
└────────────┬──────────────────┘
             ▼
┌───────────────────────────────┐
│ 4. Student forward             │  in_repo  |  hf  |  unsloth
└────────────┬──────────────────┘
             ▼
┌───────────────────────────────┐
│ 5. Loss                        │  λ_kd·KD  +  λ_ce·CE  +  λ_reg·reg
│  KD masked by:                 │     teacher_logits_mask
│                                │     teacher_confidence (Layer B field)
└────────────┬──────────────────┘
             ▼ backward()
┌───────────────────────────────┐
│ 6. Read Adam states            │  m_{t-1}, v_{t-1}    ── pre-step ──
└────────────┬──────────────────┘
             ▼
┌───────────────────────────────┐
│ 7. Compute signals per module  │  Surprise, Stability, Repetition
└────────────┬──────────────────┘
             ▼
┌───────────────────────────────┐
│ 8. R/F/P policy                │  Top-M modules → classify each
└────┬───────┬─────────┬────────┘
     ▼       ▼         ▼
   [R]     [F]       [P]                    ── outer optimizer.step ──
  store   Top-K     allow                          │
  embed   LoRA      base                           ▼
  only    write     update                  ┌───────────────────┐
                                              │ 9. Eval (every N) │  per-domain CE/PPL,
                                              │                   │  forgetting delta,
                                              │                   │  R-store hit rate,
                                              │                   │  LoRA sparsity
                                              └─────────┬─────────┘
                                                        ▼
                                              ┌───────────────────┐
                                              │ 10. Checkpoint     │  base/lora split,
                                              │   (atomic, LRU)    │  optimizer m/v,
                                              │                    │  controller EMAs,
                                              │                    │  R-store, RNG
                                              └───────────────────┘
```

## Quick start

```bash
# Core install (CPU smoke + tests; no HF / Unsloth)
pip install -e ".[dev]"

# Existing synthetic CPU smoke (5 steps, random tokens + random teachers)
python train.py --config-name pilot_smoke

# Real-data extras (HuggingFace datasets, AutoTokenizer, AutoModelForCausalLM)
pip install -e ".[dev,data]"

# Unsloth Qwen3.5 backbone (GPU)
pip install -e ".[dev,data,unsloth]"

# Tests (CPU only — every external dep is monkeypatched)
pytest tests/ -v

# Warm-up smoke on REAL HF Layer-C data (CPU, ~4 min): phased stream,
# R/F/P routing, event trace, phase-boundary eval, checkpoint + resume
python train.py --config-name stream_smoke
```

## Running experiments

The paper's experiment matrix (baselines, per-signal ablations, the
permanent-memory study, headline 1.7B runs) is driven by manifests in
`configs/experiments/` — see **[docs/experiments.md](docs/experiments.md)** for what each experiment answers, and **[docs/running.md](docs/running.md)** for every command (preflight, smoke, single-GPU, multi-GPU, resume, analysis)
for the full reproducibility guide.

```bash
pip install -e ".[data,analysis]"

# Group A: TriHOPE vs full-FT / LoRA-only / no-retrieval / no-consolidation
python scripts/run_experiment.py configs/experiments/baselines_small.yaml

# Group C: when does P fire, does it help retention, is a bad merge fatal?
python scripts/run_experiment.py configs/experiments/p_study_small.yaml

# Group B: ablate every routing signal + Top-M / Top-K
python scripts/run_experiment.py configs/experiments/ablation_grid.yaml

# Tier 4: live logit-KD robustness check (appendix)
python scripts/run_experiment.py configs/experiments/live_kd_small.yaml --concurrent 2

# Group D (optional): headline Qwen3-1.7B runs
python scripts/run_experiment.py configs/experiments/headline.yaml

# Tables + figures (Table 1/2 analogues, ablation deltas, P timelines, ...)
python -m analysis.run_report runs/baselines_small_v1
```

Every run leaves a full audit trail of *when and why parameters changed*:
`events.jsonl` (per-module routing decisions with all signal values,
consolidation events with pre-merge signals, phase boundaries),
`run_summary.json` (memory/time/retention), and a checkpointed per-module
write ledger. Runs are resumable **bit-exactly** — an interrupted+resumed
run reproduces the uninterrupted run's losses to the last bit
(`tests/test_resume_exact.py`), and the paper's formal claims (Theorems
1-2, Corollaries, all-open AdamW reduction) are enforced directly by the
test suite (`tests/test_masked_adamw.py`, `tests/test_consolidation.py`).

## Backbones

Hivemind supports three student backbones, switchable via `model.backbone`:

| Backbone | Module | What it is | When to use |
|---|---|---|---|
| `in_repo` | `student.model.StudentModel` | Hand-rolled transformer + our `LoRALinear` | CPU smoke tests, ablations, controller sanity checks |
| `hf` | `student.hf_backbone.HFStudent` | `AutoModelForCausalLM` (Qwen / Llama) + `LoRALinear` injected in-place | Vanilla HF training without Unsloth; inspectable LoRA state |
| `unsloth` | `student.unsloth_backbone.UnslothStudent` | `FastLanguageModel.from_pretrained` + PEFT LoRA via `get_peft_model` | Production training, sub-2× VRAM and ~1.5× faster than vanilla |

The controller / writer / consolidation / regularizer / evaluator all talk to LoRA through the `LoRAAdapter` protocol (`student/lora_adapter.py`). The same Top-K masking, F→P merge, and sparsity stats work whether the LoRA was attached by us or by PEFT — no isinstance branches outside the dispatch point.

## Teachers

Three teacher types via `teachers.mode`:

| Mode | Class | What it does | Vocab requirement |
|---|---|---|---|
| `synthetic` | `create_synthetic_teachers` | Frozen randomly-initialised StudentModel copies | n/a (debug only) |
| `cache` | `CacheBackedTeacher` | Pulls precomputed teacher logits from NPZ files via `TeacherLogitsCache`; if missing, KD is skipped per-row via `teacher_logits_mask` | matches the tokenizer that generated the NPZs |
| `live` | `HFTeacher` | Live `AutoModelForCausalLM.forward(tokens)`; vocab match validated at construction | must match student's `vocab_size` exactly |

Cache mode is the default for HF training because the supervision text was generated in advance (per-domain `teacher_id` values and their provenance: `docs/data_provenance.md`). No logit files are published, so cache mode is sequence-level distillation (CE on the teacher's text; the KD term is inert). `stream_small.yaml` and `stream_headline.yaml` declare four Qwen-vocabulary live teachers for `teachers.mode=live`; `configs/experiments/live_kd_small.yaml` is the logit-KD robustness tier.

## Datasets

Three "layers" per domain, exactly as `docs/dataset_preperation.md` describes:

- **Layer A** — canonical samples: `sample_id, input_text, target_text, domain, subdomain, difficulty, source, has_gold_label, bucket_id`.
- **Layer B** — teacher supervision: `sample_id, teacher_id, teacher_output_text, teacher_logits_path, teacher_confidence, teacher_entropy`.
- **Layer C** — joined training-ready view of A × B.

Hivemind's loader (`data/hf_loader.py::HivemindHFDataset`) loads `{domain}-layerA-final` and `{domain}-layerB-final` from `repo_prefix/` and **joins them on `sample_id` at load time**. Layer C is therefore optional — the loader recreates it.

Currently published under `hivemind-research/`:

```
hivemind-research/code-layerA-final
hivemind-research/code-layerB-final
hivemind-research/code-layerC-final
hivemind-research/math-layerA-final
hivemind-research/math-layerB-final
hivemind-research/math-layerC-final
```

`teacher_logits_path` is a path string like `teacher_logits/code/train/{sample_id}_code.npz`. Set `data.teacher_logits_root` to wherever you've materialised those NPZ files (local dir or fsspec URL); the loader resolves paths against it.

## Configs

Hivemind uses [Hydra](https://hydra.cc/). Configs in `configs/`:

| Config | Backbone | Teachers | Steps | Device | Purpose |
|---|---|---|---|---|---|
| `pilot_smoke` | `in_repo` | synthetic | 5 | CPU | Original smoke contract — used by tests |
| `pilot` | `in_repo` | synthetic | 50k | CUDA | Original research config |
| `pilot_hf` | `in_repo` | cache | 50k | CUDA | HF datasets + DeepSeek R1 NPZs (in-repo student keeps vocab=151,936) |
| `pilot_hf_smoke` | `in_repo` | cache | 5 | CPU | HF-path smoke (10 rows/domain, max_seq_len=128) |
| `pilot_hf_gpu` | `hf` | cache | 50k | CUDA | Qwen2.5-0.5B + DeepSeek R1 supervision; distill consolidation; eval every 500 |
| `pilot_hf_live` | `hf` | live | 50k | CUDA | Live HF teachers (Qwen-vocab only — see "live teachers" caveats) |
| `pilot_unsloth` | `unsloth` | cache | 50k | CUDA | **Recommended.** Qwen3.5-4B (or 9B/27B) + DeepSeek R1 cache + consistency penalty |

Override anything from the CLI:

```bash
python train.py --config-name pilot_unsloth \
  model.pretrained_name=unsloth/Qwen3.5-9B \
  data.batch_size=2 \
  distillation.lambda_kd=1.0 \
  data.teacher_logits_root=/mnt/teacher_logits
```

### Key config sections

```yaml
model:           # backbone, pretrained_name, dim/heads/layers, LoRA rank/alpha, target_modules
teachers:        # mode (synthetic|cache|live), teacher_ids per domain, pretrained map for live
distillation:    # tau, lambda_kd, lambda_ce, lambda_reg, use_teacher_confidence, confidence_floor
regularization:  # weight_decay, trust_region_weight, anti_forgetting_weight, lora_sparsity_weight, consistency_weight
controller:      # surprise/stability/repetition thresholds, policy top-M, consolidation strategy
data:            # source (synthetic|hf), router (cosine|metadata), domains, tokenizer, max_seq_len
optim:           # lr, betas, weight_decay (AdamW)
checkpoint:      # enabled, dir, save_every, keep_last, resume_from
eval:            # enabled, interval, max_batches, split, track_forgetting
logging:         # backend (json|wandb), path
```

## Loss

Per-row, per-step:

```
L = λ_kd · KD(p_T, p_S; τ) · m_kd · m_resp
  + λ_ce · CE(student, labels)
  + λ_reg · ( λ_wd·||θ||² + λ_tr·KL(p_old ‖ p_new)
            + λ_af·Σω(θ−θ_old)² + λ_sp·||φ_F||₁
            + λ_cons·KL(p_θ(x) ‖ p_θ(x̃)) )
```

Where:

- `m_resp` is `(labels != -100)` — KD is computed only on response positions, never prompt or pad.
- `m_kd` is per-row `teacher_logits_mask · teacher_confidence^p` — KD is suppressed for rows where the NPZ cache missed, and down-weighted for low-confidence supervision (`Layer B teacher_confidence`).
- `λ_cons` is the consistency penalty (Theory 201 §5.2): two stochastic forward passes under dropout produce `KL(p₁.detach() ‖ p₂)`; identical to dropout-as-noise data augmentation.

## Project structure

```
src/hivemind/
├── data/                     # HF + synthetic data loading
│   ├── hf_loader.py          #   HivemindHFDataset: layerA × layerB join per domain
│   ├── tokenizer.py          #   build_tokenizer(name) → AutoTokenizer wrapper
│   ├── teacher_cache.py      #   LRU NPZ reader, miss-safe (no KD on miss)
│   ├── collate.py            #   DistillCollator: input_ids/labels/teacher_logits/conf
│   ├── synthetic.py          #   SyntheticTextDataset (legacy CPU smoke)
│   └── dataloader.py         #   build_dataloader dispatcher
│
├── student/                  # Three backbones, one LoRA protocol
│   ├── model.py              #   in_repo: StudentModel (toy transformer + LoRALinear)
│   ├── hf_backbone.py        #   hf: HFStudent — AutoModelForCausalLM + LoRA injection
│   ├── unsloth_backbone.py   #   unsloth: UnslothStudent — FastLanguageModel + PEFT
│   ├── _hf_common.py         #   ArchSpec / ProjView / detect_arch (shared)
│   ├── lora.py               #   LoRALinear (our native LoRA)
│   ├── lora_adapter.py       #   LoRAAdapter Protocol + Native + PEFT adapters
│   ├── attention.py, ffn.py  #   in_repo blocks (RoPE attn, SwiGLU FFN)
│   └── config.py             #   StudentConfig, LoRAConfig
│
├── controller/               # Theory 101: gradient-based R/F/P routing
│   ├── module_index.py       #   Enumerate (layer, attn|ffn, P|F) modules
│   ├── surprise.py           #   AdamSurprise: S_t(j) = mean(g²/v)
│   ├── stability.py          #   StabilityTracker: C_t / Stab_t / V_t (+ state_dict)
│   ├── repetition.py         #   FusedRepetition: mom + bucket + retrieval
│   ├── signals.py            #   SignalComputer (aggregates trackers, state_dict)
│   ├── policy.py             #   RFPPolicy.decide → list[StoreAction]
│   ├── writer.py             #   WriteExecutor: per-action R/F/P side effects
│   ├── consolidation.py      #   ConsolidationScheduler (direct + distill strategies)
│   └── config.py             #   ControllerConfig dataclass
│
├── stores/
│   ├── retrieval.py          # R-store: FIFO embedding deque, cosine query, state_dict
│   ├── fast.py               # F-store: Top-K mask via LoRAAdapter
│   └── permanent.py          # P-store: F→P merge via LoRAAdapter
│
├── teacher_registry.py       # TeacherRegistry, CacheBackedTeacher, factories
├── teacher_router.py         # TeacherRouter (cosine), MetadataRouter, batch_teacher_forward
├── teacher_hf.py             # HFTeacher (live AutoModelForCausalLM, vocab-checked)
├── distillation.py           # KD + CE + combined objective + masked KD
├── regularization.py         # Weight decay, trust region, AF, LoRA sparsity, consistency
├── embedding.py              # SharedEmbedding (mean-pooled tokens)
├── checkpoint.py             # CheckpointManager: atomic save, base/lora split, RNG, resume
├── evaluation.py             # run_evaluation: per-domain loss/PPL, forgetting tracker
├── training.py               # 13-step training loop
└── optim/
    ├── masked_adamw.py       # AdamW exposing m/v + selective masking
    └── factory.py            # build_optimizer
```

## Consolidation

Two F→P strategies via `controller.consolidation.merge_strategy`:

- `direct` — `base.weight += B @ A · (α/r)` then reset LoRA. Fast, lossless, default.
- `distill` — capture student-with-LoRA logits on a replay batch, zero LoRA via `_lora_zeroed` context, train base for `distill_iters` AdamW steps to match the captured target via KL, reset LoRA. Soft consolidation; preserves base generalization structure.

The replay buffer is automatically populated by `consolidator.record_batch(tokens)` in the training loop. Falls back to direct merge if the buffer is empty.

## Evaluation

`evaluation.run_evaluation` runs at `eval.interval` step boundaries when `eval.enabled` is true. Per-domain validation dataloaders are built at startup (one per domain in `data.domains`) and produce metrics keyed `eval/<domain>/<metric>`:

- `loss`, `ppl`, `tokens`, `samples`
- `bucket/<bucket_id>/loss` — top-5 worst buckets per domain
- `forgetting_delta` — current loss minus first-eval baseline (positive = catastrophic forgetting on an old domain)
- `lora/sparsity_mean`, `lora/sparsity_max` — fraction of LoRA entries below threshold
- `retrieval/buffer_size`, `retrieval/hit_rate`
- `macro_loss`, `macro_ppl` — domain-averaged

`ForgettingTracker` is checkpointed alongside the model so the baseline survives resume.

## Checkpointing

`CheckpointManager` (in `checkpoint.py`) persists everything needed for bit-exact resume:

- Student state, split into `base.pt` (P-store) and `lora.pt` (F-store) files
- Optimizer state (Adam `m, v, step` per parameter)
- Controller state (per-module stability EMAs, repetition bucket counters, retrieval hit buffer)
- R-store contents (deque of `RetrievalEntry`)
- Consolidator replay buffer
- `ForgettingTracker` baseline
- RNG state (python, numpy, torch CPU + CUDA)

Atomic write via tmp directory + rename. `keep_last` LRU pruning. Resume via `checkpoint.resume_from: latest | <step> | <abs path>`. JSON `meta.json` records step + timestamp.

## Theoretical foundation

Two interconnected papers:

- **Theory 201 — Distillation block** ([`docs/theory_201.md`](docs/theory_201.md)). Multi-teacher routing via cosine similarity in a shared embedding space. Teacher / student temperature softmaxes. KD + CE + regularization objective. Backpropagation. Five regularization forms — weight decay, trust-region KL-to-old, anti-forgetting, LoRA sparsity, dropout-consistency penalty.
- **Theory 101 — Controller block** ([`docs/theory_101.md`](docs/theory_101.md)). Module index `j = (layer, block, P|F)`. Adam-based surprise. Stability via directional cosine, Adam ratio, windowed variance. Repetition fusion: momentum + bucket-surprise decay + retrieval hit rate. Top-M selection. R/F/P decision policy. Write execution: R (store, no update), F (Top-K LoRA on rank components or rows), P (consolidate). Periodic F→P consolidation (direct or distillation-based).

The key insight: Adam's moment estimates aren't just optimization machinery — they're a compressed per-parameter learning history. `m` is the running direction; `v` is the running variance; `m²/v` is signal-to-noise. Hivemind reads them as the controller's primary observation.

The controller's R/F/P decisions exist to provide experimental evidence that those signals are informative: if routing decisions based purely on optimizer states improve distillation outcomes vs. uniform updates, the signals must contain real, actionable information about *what* the model is learning and *where* it should be remembered.

## Testing

```bash
pytest tests/ -v
```

All tests are CPU-only, deterministic, and self-contained — `transformers`, `unsloth`, `datasets`, and `peft` are monkeypatched in the relevant tests so nothing downloads or hits the network.

| Test file | Coverage |
|---|---|
| `test_lora.py` | LoRALinear merge / reset / Top-K mask |
| `test_lora_adapter.py` | LoRAAdapter protocol over native + fake-PEFT shapes |
| `test_student_model.py` | Forward, backward, parameter group split |
| `test_hf_student.py` | HFStudent injection over a fake HF model |
| `test_unsloth_student.py` | UnslothStudent over a fake `unsloth.FastLanguageModel` |
| `test_module_index.py` | (layer, block, P\|F) enumeration |
| `test_surprise.py`, `test_stability.py`, `test_repetition.py` | Each signal vs hand-computed values |
| `test_policy.py` | R/F/P decision boundaries |
| `test_writer.py` | R / F / P side effects on grads + stores |
| `test_consolidation.py` | F→P direct merge |
| `test_consolidation_distill.py` | F→P distill strategy + replay fallback |
| `test_distillation.py` | KD loss correctness, tau scaling |
| `test_regularization.py` | Each reg term in isolation |
| `test_consistency_penalty.py` | Theory 201 §5.2 — zero when deterministic, positive under dropout, label-mask honored |
| `test_confidence_gating.py` | KD scales with `teacher_confidence`, `confidence_floor` |
| `test_hf_loader.py` | layerA × layerB join, missing-A handling |
| `test_teacher_cache.py` | NPZ hit/miss/LRU |
| `test_collate.py` | Padding, attention mask, label `-100`, KD mask |
| `test_metadata_router.py` | teacher_id → index, domain fallback, miss count |
| `test_teacher_router.py` | cosine routing |
| `test_hf_teacher.py` | HFTeacher vocab-mismatch raise, frozen params |
| `test_evaluation.py` | per-domain loss / forgetting delta / hit rate |
| `test_checkpoint.py` | save/load roundtrip, latest pointer, keep_last pruning |
| `test_training_loop.py` | End-to-end synthetic 3-step |
| `test_training_loop_hf.py` | End-to-end HF-style dict batches |
| `test_determinism.py` | Same seed → same losses |

## Logging

Each step emits:

- **Loss components** — `loss/kd`, `loss/ce`, `loss/reg`, `loss/total`, `loss/kd_row_hit_frac`, `loss/teacher_conf_mean`
- **Signals** — `signals/surprise_mean`, `signals/stability_C_mean`, `signals/repetition_mean`
- **Routing** — `write/r_count`, `write/f_count`, `write/p_count`
- **Stores** — `retrieval/buffer_size`, `consolidation/count`
- **Eval (every `eval.interval`)** — `eval/<domain>/{loss,ppl,forgetting_delta}`, `eval/lora/sparsity_*`, `eval/retrieval/hit_rate`, `eval/macro_*`

Backends: JSON (default, `logging.path`) or Weights & Biases (`logging.backend: wandb`).

## Recommended pilot

For a real first run on a single GPU:

```bash
pip install -e ".[dev,data,unsloth]"

python train.py --config-name pilot_unsloth \
  data.teacher_logits_root=/path/to/teacher_logits \
  distillation.lambda_kd=1.0
```

Expected over the first 1–2 k steps:

- `loss/total` falls monotonically; `eval/<domain>/loss` decreases at the eval interval.
- `write/r_count`, `write/f_count`, `write/p_count` are all non-zero (controller is routing).
- `signals/repetition_mean` rises as bucket statistics accumulate.
- After `controller.consolidation.period` steps, `consolidation/count > 0`.
- VRAM ≈ 10 GB for Qwen3.5-4B at `batch_size=1, max_seq_len=2048`; bump `model.pretrained_name` to `unsloth/Qwen3.5-9B` (22 GB on H100) or `Qwen3.5-27B` (56 GB on A100-80GB) without code changes.

## License

Apache 2.0

# Hivemind

**Multi-Teacher Knowledge Distillation with Tri-Store (R/F/P) Memory Routing**

Hivemind explores a central research question: *can the internal states of adaptive optimizers serve as an interpretable window into LLM learning dynamics?* We formalize three signal families — **surprise**, **stability**, and **repetition** — derived from Adam's moment estimates, and demonstrate their utility by building a knowledge routing system that automatically directs learned knowledge to the appropriate memory store during multi-teacher distillation.

## Core Idea

During training, Adam maintains per-parameter exponential moving averages of gradients (m) and squared gradients (v). These are typically treated as optimization bookkeeping. We show they encode rich, interpretable information about *what the model is learning*:

| Signal | What it reads | What it means |
|--------|--------------|---------------|
| **Surprise** | g²/v — current gradient vs Adam's expected scale | "Is this input teaching something unexpected?" |
| **Stability** | cos(g, m) — gradient alignment with momentum | "Is this module learning consistently or thrashing?" |
| **Repetition** | EMA of alignment + bucket surprise decay | "Has this pattern appeared before, and is the model getting better at it?" |

These signals drive a **tri-store routing policy** that decides, per module, where knowledge should be stored:

- **R-store (Retrieval)** — novel one-offs → store in external memory, no weight update
- **F-store (Fast/LoRA)** — recurring but unstable → learn quickly in reversible adapters
- **P-store (Permanent)** — recurring and stable → consolidate into base weights

## Architecture

```
Input batch
    │
    ▼
┌─────────────────────┐
│  Shared Embedding    │  f_emb(x) → routing vector
│  (mean-pooled)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Teacher Router      │  cosine similarity → argmax → select teacher
│  (prototype-based)   │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Teacher │ │Student │
│forward │ │forward │
│(frozen)│ │        │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────────┐
│  Loss Computation    │  L = λ_KD·L_KD + λ_CE·L_CE + λ_reg·L_reg
└────────┬────────────┘
         │
         ▼  backward()
┌─────────────────────┐
│  Read Adam States    │  m_{t-1}, v_{t-1} per module (BEFORE optimizer.step)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Signal Computation  │  Surprise, Stability, Repetition per module
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  R/F/P Policy        │  Top-M modules → classify → route
└────────┬────────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
   [R]  [F]  [P]
  store  LoRA  base
  embed  Top-K  weight
  only   write  update
```

## Project Structure

```
src/hivemind/
├── student/            # Transformer + LoRA (P-store = base, F-store = adapters)
│   ├── model.py        # StudentModel with tied embeddings
│   ├── attention.py    # Causal self-attention with RoPE + LoRA
│   ├── ffn.py          # SwiGLU FFN + LoRA
│   └── lora.py         # LoRALinear: merge/reset/Top-K masking
│
├── controller/         # Theory 101: gradient-based R/F/P routing
│   ├── surprise.py     # S_t(j) = mean(g²/v) — Adam-based novelty
│   ├── stability.py    # C_t (directional), Stab_t (Adam ratio), V_t (variance)
│   ├── repetition.py   # R_mom, R_hash (bucket decay), R_ret (embedding hits)
│   ├── policy.py       # R/F/P decision logic per module
│   ├── writer.py       # Execute write actions on model + stores
│   └── consolidation.py# Periodic F→P merge
│
├── stores/             # Memory stores
│   ├── retrieval.py    # R-store: FIFO embedding buffer
│   ├── fast.py         # F-store: LoRA management + Top-K
│   └── permanent.py    # P-store: base weight consolidation
│
├── embedding.py        # Shared routing embedding
├── teacher_registry.py # K frozen teachers + prototypes
├── teacher_router.py   # Cosine similarity routing
├── distillation.py     # KD + CE + combined objective
├── regularization.py   # Weight decay, trust-region, anti-forgetting, sparsity
├── training.py         # Full 13-step training loop
└── optim/
    └── masked_adamw.py # AdamW with exposed m/v states + gradient masking
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run smoke test (CPU, 5 steps, synthetic data)
python train.py --config-name pilot_smoke

# Run tests
python -m pytest tests/ -v

# GPU training (requires CUDA)
python train.py --config-name pilot
```

## Configuration

Hivemind uses [Hydra](https://hydra.cc/) for configuration. Two presets are provided:

| Config | Purpose | Model | Steps | Device |
|--------|---------|-------|-------|--------|
| `pilot_smoke` | Smoke test | 64d, 2L, 4H | 5 | CPU |
| `pilot` | Research | 512d, 12L, 8H | 50k | CUDA |

Override any parameter from the command line:

```bash
python train.py --config-name pilot train.steps=1000 model.dim=256 distillation.tau=2.0
```

### Key Configuration Sections

```yaml
model:           # Student architecture (dim, layers, heads, LoRA rank)
teachers:        # Number of teachers, prototype seed samples
distillation:    # Temperature (tau), loss weights (lambda_kd, lambda_ce, lambda_reg)
regularization:  # Weight decay, trust-region, anti-forgetting, LoRA sparsity
controller:      # Signal thresholds, policy params, consolidation schedule
data:            # Source, vocab size, sequence length, batch size
optim:           # Learning rate, betas, weight decay
```

## Training Pipeline

Each training step follows 13 stages, with a critical ordering constraint:

1. **Embed** samples via shared embedding
2. **Route** each sample to best teacher (cosine similarity)
3. **Teacher forward** (frozen, grouped by assignment)
4. **Student forward**
5. **Compute loss** (KD + CE + regularization)
6. **Backward** pass
7. **Read Adam states** (m_{t-1}, v_{t-1}) — *must happen before optimizer.step()*
8. **Compute signals** (surprise, stability, repetition per module)
9. **R/F/P policy** (select top-M modules, classify each)
10. **Execute writes** (R: store + zero grad, F: Top-K LoRA, P: allow base update)
11. **Optimizer step** (on remaining parameters)
12. **Consolidation check** (periodic F→P merge)
13. **Log** metrics and signal distributions

## Theoretical Foundation

The system implements two interconnected theory documents:

**Theory 201 — Distillation Block:**
Multi-teacher routing via cosine similarity in a shared embedding space, producing a combined training objective L = λ_KD·L_KD + λ_CE·L_CE + λ_reg·L_reg. This objective generates the gradient that the controller reads.

**Theory 101 — Controller:**
After backpropagation, the controller reads per-module gradient statistics and Adam optimizer states to compute three signal families. These signals determine where knowledge is stored — a decision that is made *per module, per training step*, without any external classifier or auxiliary model.

The key insight: Adam's moment estimates are not just optimization machinery. They are a compressed, per-parameter history of what the model has learned (m — direction) and how volatile that learning has been (v — scale). The ratio m²/v directly estimates signal-to-noise per parameter — a free interpretability signal.

## Metrics & Logging

Training logs include:

- **Loss components:** KD loss, CE loss, regularization loss, total loss
- **Signal distributions:** mean surprise, stability (C_t), repetition across modules
- **Routing actions:** R/F/P counts per step
- **Store state:** retrieval buffer size, consolidation events

Logging backends: JSON (default) or Weights & Biases.

## Testing

```bash
python -m pytest tests/ -v
```

All tests use tiny dimensions (vocab ≤ 256, dim ≤ 64, 1-2 layers), CPU only, deterministic seeds. Test coverage includes:

- LoRA forward/merge/reset/Top-K masking
- Student model shapes and parameter group separation
- KD loss correctness (manual computation, tau scaling)
- All three signal families against hand-computed values
- R/F/P policy boundary conditions
- Write executor (R-store buffering, F-store masking, P-store flow)
- F→P consolidation correctness
- End-to-end training loop (3-step smoke)
- Determinism (same seed → same losses)

## Research Context

This work explores optimizer states as an underexploited source of interpretable training signals in LLMs. Rather than inspecting model outputs or activations at inference time, we read the optimization process itself to understand *how* the model is learning — which modules are surprised, which are stable, and which knowledge patterns are recurring.

The R/F/P routing system serves as experimental evidence that these signals contain real, actionable information about learning dynamics. If routing decisions based purely on optimizer states lead to better distillation outcomes, the signals must be genuinely informative.

See [docs/research_framing.md](docs/research_framing.md) for the full research narrative.

## License

Apache 2.0

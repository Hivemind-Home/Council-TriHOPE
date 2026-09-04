# Hivemind

Multi-Teacher Knowledge Distillation with Tri-Store (R/F/P) Memory Routing.

## Quick Start

```bash
pip install -e ".[dev,data]"
python -m pytest tests/ -q                        # 499 tests, CPU-only
python -m hivemind preflight --config-name stream_small --metadata-only
python train.py --config-name stream_smoke        # CPU, real data, ~5 min
```

Full command reference: **docs/running.md**. What each experiment answers:
**docs/experiments.md**.

## Architecture

The codebase implements two interconnected theory documents:

- **Theory 201** (distillation block): Multi-teacher routing via cosine similarity in shared embedding space → KD + CE + regularization loss
- **Theory 101** (controller): Gradient-based R/F/P routing using Adam optimizer states (surprise, stability, repetition signals)

### Key directories

- `src/hivemind/student/` — Transformer model with LoRA adapters (P-store = base weights, F-store = LoRA)
- `src/hivemind/controller/` — Signal computation (surprise, stability, repetition) and R/F/P policy
- `src/hivemind/stores/` — Retrieval (R), Fast (F), Permanent (P) memory stores
- `src/hivemind/optim/` — MaskedAdamW with exposed m/v states
- `configs/` — Hydra YAML configs (pilot_smoke for CPU, pilot for GPU)

### Training pipeline (per step)

1. Embed samples → 2. Route to teacher (cosine similarity) → 3. Teacher forward → 4. Student forward → 5. Combined loss → 6. Backward → 7. Read Adam m/v states → 8. Compute signals → 9. R/F/P policy → 10. Execute writes → 11. Optimizer step → 12. Consolidation check → 13. Log

### Critical ordering constraint

Steps 7-8 (read `m_{t-1}`, `v_{t-1}`) MUST happen AFTER backward but BEFORE optimizer.step() which updates m/v.

## Conventions

- Python 3.10+, PyTorch 2.4+
- `src/` layout with hatchling build
- Hydra configs in `configs/`
- All tests use tiny dims (vocab≤256, dim≤64), CPU only
- Reference codebase: `nested_learning/` (HOPE architecture)

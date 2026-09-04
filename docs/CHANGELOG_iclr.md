# ICLR-2027 push — changelog

One entry per task from `docs/TriHOPE_Agent_Task_Document.md`. Each entry
records what changed, which files, which tests, which config keys, and any
deviation from the task document with the reason.

## T0 — environment and baseline sanity

- Environment: `/root/venv` (Python 3.10.12, torch 2.13.0+cpu, transformers
  5.14.1, datasets 5.0.1, pandas 2.3.3, matplotlib 3.10.9, pytest 9.1.1,
  ruff 0.16.2). `pip install -e ".[dev,data,analysis]"` already satisfied.
- `pytest tests/ -q`: all green — 420 passed (402 test functions across 45 files, some parametrised), 2
  GPU-only skips (`tests/test_audit_fixes.py`). `ruff check src tests
  analysis scripts`: clean.
- `stream_smoke` / resume / preflight: recorded under the tasks that first
  exercise them (T2 for the smoke + resume, T8 for preflight on every
  manifest) because they need the HF download; see those entries.
- Baseline commit: `8dc4917` (C18). No `runs/` exist.

## T1 — sustained stability everywhere, and make P fire

- `PolicyConfig.stability_source: instant|sustained` (validated in a new
  `__post_init__`); `RFPPolicy._classify` reads `stability_C_sustained` for
  the P test when `sustained`. R/F logic untouched. Neutral pinning under
  the cosine ablation already set both fields (`signals.py`), verified.
- `run_config` event now records `policy.stability_source` and
  `consolidation.stability_mode`.
- Configs: `stream_small.yaml` and `stream_headline.yaml` set
  `policy.stability_source: sustained`, `consolidation.stability_mode:
  sustained`, `period: 250`, `min_stability_C: 0.4`, `min_repetition: 0.4`,
  explicit `stability.c_ema_alpha: 0.1`. `stream_smoke.yaml` unchanged.
  `p_study_small.yaml` drops the three overrides that are now defaults.
- `ablation_grid.yaml`: `stability_instant`, `consolidation_strict`.
- Tests: `tests/test_policy.py` (+3: sustained → P / instant → F on the
  same signals; R unaffected by the source; invalid value raises).
- No deviation.

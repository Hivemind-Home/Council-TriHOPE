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

## T2 — R → F replay promotion (deferral, not deletion)

- Config: new `RetrievalConfig` (`controller.retrieval`): `replay_on_hit`
  (default false → R stays write-only), `hit_threshold` (distinct earlier
  R steps a bucket needs), `replay_batches`, `replay_target` (only `F`).
- `RetrievalEntry` is now `eq=False` (identity equality; the generated
  field-wise `__eq__` raises on multi-element tensors) and carries the
  representative row when replay is on: `input_ids`, `labels`,
  `kd_row_mask`, `teacher_confidence`. `RetrievalStore` gains
  `entries_for_bucket`, `distinct_steps_for_bucket`, `pop_for_bucket`
  (step-grouped FIFO: the writer appends one entry per R action, so up to
  Top-M identical entries per step count as one recurrence) and a
  checkpointed `replayed_total`. Old `stores.pt` files load unchanged.
- `WriteExecutor.execute(..., sample=None)` attaches the row to R entries.
- Loop (`training.py`): `_capture_replay_sample` takes row 0 (broadcast
  from rank 0 every step under DDP — a collective must not sit behind a
  per-rank branch); the replay micro-step runs after Step 11 inside the
  same step index: forward/backward on the parked row (B=1), Top-M F-type
  modules by raw grad norm via the new pure `RFPPolicy.select_top_m` (no
  stateful tracker sees the replay), Top-K LoRA masks through the writer,
  a second masked optimizer step. Emits `replay` events (distinct from
  `decision`, so per-step decision counts and action shares are untouched),
  `write/replay_count`, `run_summary.extra.retrieval.replayed_total`, and
  a second `assert_rank_consistent` payload with the replayed decisions.
- Tests: `tests/test_replay.py` (+9: step-keyed hits/pop/eviction,
  state-dict round-trip incl. legacy entries, identity equality, kd-mask
  capture, replay fires at step 1 for a row parked at step 0 with all-F
  actions and no stream step lost, flag off is bit-identical to a config
  without the key, replay changes the trajectory only after it lands);
  `tests/test_resume_exact.py` (+1: resume after a replay is bit-exact,
  checkpoint carries `replayed_total`).
- **Deviations from the task document.** (1) The replay is an extra
  micro-step *inside* the triggering step, not a substitution of the next
  stream batch: substitution breaks `step == stream index`, on which the
  phase map, eval schedule, `force_consolidate_steps`, checkpoint cadence,
  the digest guard and DDP sharding all rely, and would require growing
  `train.steps`. Nothing new is checkpointed beyond the R-store. (2) The
  trigger counts distinct origin *steps*, not entries. (3) The replayed
  KD term is CE-only for HF rows (cached logits are not stored). (4) A
  bucket-keyed trigger can never fire on `novel_inject` batches (each has
  a unique synthetic bucket); on the real stream replay fires on
  recurrent-phase rows that were first routed R.

## T3 — baseline controllers for E1

- `PolicyConfig.mode: rfp | surprise_only | adam_score` (+ `adam_score_high`,
  default 0.5; validated). `surprise_only` = Titans-style gate (R iff
  S ≥ surprise_high, F otherwise, never P); `adam_score` = MoLF-style
  two-tier routing on the Adam SNR m²/v already computed as
  `stability_adam` (P iff ≥ threshold, else F, never R). `disable_stores`
  still applies in every mode.
- `WriterConfig.flag_p_for_consolidation` (default true): with `false` a
  P action on an F-type module opens the adapter but never flags it —
  MoLF has no merge path.
- `ConsolidationConfig.trigger: signals | plateau`, `plateau_window=200`,
  `plateau_tolerance=0.01`, `plateau_min_gap=200`. New `PlateauDetector`
  (owned by `ConsolidationScheduler`, serialised under
  `consolidation.pt["plateau"]`): plateau = first-half vs second-half
  window means improve by less than the tolerance; the window is cleared
  on a fire and `min_gap` blocks refires. `should_check` is false under
  the plateau trigger (no signal sweep). Loop call site sits outside the
  `controller_enabled` gates (the spec runs with the controller off and
  every adapter open); every rank feeds its own loss, rank 0's verdict is
  agreed through `all_reduce_max_int`, the window is reset on all ranks.
  `consolidation` events now carry `trigger: signals | plateau | forced`.
- `DebugConfig.policy_override="random_matched"`, `random_shares_path`,
  `random_shares_spec_id="trihope"`, `random_unit: module | step`.
  `RFPPolicy` takes `debug=` and `seed=`; `decide(..., step=, phase=)`.
  Shares come from `action_share_by_phase.csv` (`*_share_mean` columns,
  filtered to the spec, renormalised); draws use a local
  `torch.Generator` seeded from `(seed, step[, module rank])` — identical
  on every rank, no global-RNG use; missing file/phase → uniform thirds
  with one warning; `disable_stores` honoured.
- `run_config` event records `policy_mode`, `policy_override`,
  `adam_score_high`, `consolidation.trigger` and plateau parameters.
- Tests: `tests/test_baseline_policies.py` (+14: label sets per mode,
  adam_score + writer flag never flags consolidation, random_matched
  frequencies within 3 points of the file per phase / identical across
  instances / global RNG untouched / uniform fallback with warning /
  disabled stores + per-step draw, plateau detector fires on flat not on
  decreasing series with min_gap and state round-trip, invalid trigger
  raises, end-to-end plateau merges with the controller off).
- Notes / deviations: the `molf_style` spec must also set
  `controller.ablation.use_teacher_confidence=false` — otherwise
  TriHOPE's confidence gate would demote MoLF's P to F on low-confidence
  medical rows (done in the T8 manifests). MoLF's exact EPD score is
  checked in T9; `stability_adam` is the mean m²/v, documented there.
- Commit note: T2 and T3 landed in one commit (`T2+T3`) because both
  touch the same hunks of `controller/config.py`, `policy.py`,
  `writer.py` and `training.py`; the changelog entries above are still
  separate. Every later task has its own commit.
- T2 acceptance on real data (CPU, `stream_smoke ++controller.retrieval.
  replay_on_hit=true`, 70 steps, 186 s): 280 decisions (R 213 / F 62 /
  P 5), 22 parked rows replayed (88 module-level F writes) at steps
  34–39 (math_recurrent) and 53–69 (medical_recurrent / math_revisit),
  origins 4–51; zero replays in `novel_inject`, as predicted. Final loss
  4.8209 vs the 5.0543 reference run without replay in
  `docs/experiments.md`.
- T0.4 / T2 resume acceptance on real data: resuming the replay-enabled
  `stream_smoke` run from `checkpoint.resume_from=50` finishes at
  `4.8209028244018555`, identical to the uninterrupted run to the last
  digit (the checkpoint carried the popped R-store and `replayed_total`).

## T7 — budget accounting, Pareto figure, multi-experiment report

- `ModuleLedger`: per-module entries now come from a factory (no shared
  template) and gain `p_coords_opened` (P actions on base weights),
  `merged_coords` (base numel rewritten by each merge), `replay_count`;
  run-level counters `steps`, `coords_opened` (Σ open indexed coordinates
  per step — masks, or every trainable indexed coordinate when masking is
  off), `unmasked_base_coords` (full-FT's permanent writes). All
  checkpointed in `ledger.pt`; old ledgers load with zeros.
- `training.py`: `ledger.record_step(...)` every trained step;
  `_trace_merges` charges the merged block's base numel and puts
  `merged_coords` on the `consolidation` event; `run_summary.json` gains
  `permanent_writes {p_action_coords, merged_coords,
  unmasked_base_coords_total, total}`, `indexed_coords`,
  `active_fraction_mean` (additive kwargs on `write_run_summary`).
- `eval.interval_by_phase: {phase: steps}` (pure function of (step,
  phase), rank-agreeing); `stream_small.yaml` sets `{code_revisit: 50}`.
- `analysis/tables.py`: `threshold_tag` (operating point from the
  `run_config` event, not the spec id), `steps_to_recover` (first in-phase
  eval within 5 % of the own-phase loss; `None` if never), `budget_curve`
  (per (spec, tag), mean±std over seeds: permanent writes, active
  fraction, worst/mean retention delta, final per-domain loss,
  steps-to-recover, replayed_total). `analysis/figures.py`:
  `pareto_figure` (symlog x — LoRA-only sits at zero writes; two panels:
  worst retention delta and mean final loss). `analysis/run_report.py`
  accepts several experiment dirs and prefixes spec ids with the
  experiment name (`e1/trihope`) so E1 and the sweep share one figure.
- Bug fixed on the way: `action_share_by_phase`, `p_selection_stats`,
  `plot_action_composition`, `plot_p_timeline` crashed on runs with no
  `decision` events (every controller-off run: full_ft, lora_only,
  plateau_trigger). They now skip such runs.
- Tests: `tests/test_budget_curve.py` (+7).
- Deviation: `permanent_writes` is defined as (coordinate, step) write
  events on base weights, so full-FT's count is trainable-base-numel ×
  steps rather than "number of P actions" — the doc left the unit open,
  and this is the only definition under which full FT, LoRA-only and the
  merge-based controllers sit on one axis.

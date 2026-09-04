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

## T8 — manifests, shared-GPU runner mode, and the GPU runbook

- `configs/experiments/baselines_small.yaml` (E1, 3 seeds): trihope
  (replay on), **trihope_no_hash** (label-free recurrence: bucket-id
  counter off), full_ft, lora_only, no_retrieval, no_consolidation,
  surprise_gate, molf_style, plateau_trigger, random_routing, gold_ce.
  `r_tier_small.yaml` (E2, 3 seeds): trihope_replay, trihope_r_terminal,
  trihope_no_hash_replay, fp_only, p_only. `budget_sweep_small.yaml` (E3,
  1 seed, 21 points): trihope 3×3 (S × R_low), trihope_no_hash ×3,
  surprise_gate ×3, plateau_trigger ×3 tolerances, molf_style ×3 scores.
  `bad_teacher_small.yaml` lands with T5 (its keys come from T4–T6).
- `scripts/run_experiment.py --concurrent N`: N runs at a time on the one
  visible GPU without device pinning (the 0.6B student uses a fraction of
  a 96 GB card); exclusive with `--parallel-gpus` / `--nproc-per-node`.
- `docs/GPU_RUNBOOK.md`: the tiered campaign (E1 → E5 → E2/E3/E4 →
  optional 1.7B), exact commands, artifacts, the go/no-go check.
- Tests: `tests/test_run_experiment.py` (+2; manifest parse list extended).
- Deviations: `p_only` is `debug.policy_override=always_p` (every selected
  block opens fully and every adapter is flagged for the next sweep) — the
  doc's "adam_score + always open" would still be two-tier. `surprise_gate`
  and `random_routing` run with replay on so their R tier means the same
  thing as trihope's. `trihope_no_hash` is added to E1/E2/E3 as a headline
  candidate (see the plan: the bucket-id counter is dataset metadata).

## T4 — corrupted-teacher stream

- New `src/hivemind/data/corruption.py`: `CorruptionSpec`
  (`data.corrupt_teacher: {enabled, domain, phase, fraction, mode:
  shuffle|degrade, confidence, tag}`), `degrade_text` (first 25 % of
  whitespace tokens + "Final answer: <last number + 1>", or a fixed wrong
  suffix), `CorruptedTeacherView` (a `Dataset` view rewriting
  `teacher_output_text`, `teacher_id = tag`, `teacher_confidence` when
  set, `teacher_logits_path = None`, `corrupted = True`; delegates every
  other attribute to the wrapped dataset).
- `StreamSchedule(corruption=)`: selection in `_build_corruption` with
  `_rng(phase_index, tag=99)`, exposed as `corrupted_indices`,
  `corruption_partner`, `corrupted_rows_by_phase`, `corruption_plan()`,
  `is_corrupted_step()`. `config_digest` gains a `corruption` key **only
  when enabled**, so every clean checkpoint keeps its digest.
- `build_dataloader` wraps the TRAIN dataset in the stream branch after the
  schedule is built; eval and probe loaders stay clean.
- `create_hf_cache_teachers(..., extra=[(tag, domain)])` registers the
  tag as a fifth cache teacher; `training.py` builds `domain_to_index`
  first-wins (it was last-wins, which would have routed unknown-id math
  rows to the corrupted teacher). `phase_start` events carry
  `corrupted_rows`; `run_config` and `run_summary.extra` carry the block
  plus `corrupted_rows_by_phase`.
- Preflight: enabled-only checks (domain declared, phase exists and serves
  the domain, tag does not collide with a real teacher id).
- `stream_small.yaml` ships the block with `enabled: false`.
- Tests: `tests/test_corrupt_teacher.py` (+15).
- **Deviations.** (1) Selection is batch-coherent — `round(fraction ×
  num_buckets)` whole bucket subsets plus `round(fraction × other steps)`
  whole background/novel steps — not per-row: with batch_size 2 and bucket
  rows served ~19× each, per-row selection would alternate clean/corrupt
  across visits and mix within a batch, so the representative-row teacher
  label and every merge attribution would be ~50 % wrong. "Exact fraction"
  therefore holds at step level within one bucket's share. (2) Corruption
  is applied by global row index through a dataset view instead of at
  collate time (no `global_index` plumbing). A corrupted background row
  that a later mixed phase re-draws stays corrupted; the count is reported
  per phase (`corrupted_rows_by_phase`) — ≈1 row for `stream_small`.
  (3) `confidence` defaults to `null` (keep the row's own): the doc's 0.2
  would make E5 measure the confidence gate rather than the routing; 0.2
  is the separate `trihope_lowconf` arm in `bad_teacher_small.yaml` (T5).

## T5 — teacher attribution, ledger completeness, pre-merge tagging, selective rollback

- `ModuleLedger` (`tracing.py`): per-module `attribution {teacher: coords}`
  accumulated by `record_action(..., teacher=)` for every F/P/replay
  write (pending evidence since the last merge for F modules; cumulative
  direct base writes for P modules); `record_consolidation(...)` returns
  and clears it and charges each teacher its share of `merged_coords`;
  per-teacher totals (`teacher_summary()`, written to
  `run_summary.ledger_totals.teachers`); `pending_attribution`,
  `attribution_share`, `clear_pending`. Entries come from factories, so
  the nested dicts never alias; legacy ledgers load clean.
- `consolidation` events carry `attribution` and `attribution_share`;
  `pre_merge` checkpoints carry `pending_modules` + their attribution in
  `meta.json`. Every decision's teacher is the rank-synced representative
  teacher (`DistContext.broadcast_int`, inert at world size 1) and is part
  of the tier-1 rank-consistency payload.
- `DebugConfig.block_p_for_teachers` / `block_min_share`: a listed
  teacher's P actions are demoted to F (also under overrides); on resume
  every adapter whose pending attribution is dominated by a listed teacher
  is reset (`reset_lora`, optimizer state zeroed, flag dropped, attribution
  cleared, `lora_a` broadcast under DDP) with a `rollback_reset` event; the
  signal sweep refuses blocked modules (`ConsolidationScheduler.blocked`,
  `unflag`).
- `checkpoint.save_before_phases: [phase]` → tagged `pre_phase_<name>`
  save at the last step before the phase; `train.skip_step_ranges:
  [[lo, hi]]` → the sampler does not yield those steps and the loop emits
  `skipped_step reason=skip_range` (no forward, no eval, no checkpoint);
  stored in checkpoint `extra`, warned on mismatch, digest unchanged.
- `analysis.tables.teacher_attribution` / `containment`,
  `analysis.figures.containment_bars`, wired into `run_report`.
- `scripts/rollback_teacher.py`: selective rollback (first merge with
  `attribution_share[teacher] ≥ --min-share` → its `pre_merge` dir →
  resume the original Hydra overrides with the block list) and
  `--baseline full_restore` (the `pre_phase_<phase>` dir + the phase
  skipped); writes `rollback_summary.json` comparing final losses and
  retention against the original run. `--dry-run` prints the plan.
- `configs/experiments/bad_teacher_small.yaml` (E5, 3 seeds): trihope,
  trihope_nogate, trihope_lowconf (confidence 0.2), full_ft, lora_only,
  molf_style — all with the corrupted stream, `checkpoint_before_merge`,
  `keep_tagged=0`, `save_before_phases=[math_recurrent]`;
  `gradient_routing` is added by T6. `run_experiment.expand_matrix`
  flattens YAML-anchor override lists.
- Tests: `tests/test_attribution.py` (+4), `tests/test_ledger_completeness.py`
  (+2: support(θ_T − θ_0) over controller-indexed params ⊆ ∪ staged
  masks; after a forced merge, changes outside masks only on merged
  blocks), `tests/test_rollback.py` (+8), manifest parse list extended.
- **Deviations.** (1) Rollback also resets the blocked teacher's dominated
  adapters on restore — demoting P→F alone would leave its F-writes in
  adapters that a later clean P action merges wholesale. (2) Skipped
  steps skip eval too, so the skipped phase's own-phase loss is not
  recorded (the full-restore counterfactual never trains on it; retention
  on the other domains is what the comparison reads). (3) Merges without
  attribution (empty adapters) are ignored by the rollback point search.
  (4) Ledger completeness is asserted as ⊆ (a coordinate opened with an
  exactly-zero update does not move) plus the merged-block exception.

## T6 — gradient-routing baseline (E5)

- `PolicyConfig.mode="teacher_partition"`: every F-type module gets an F
  action carrying `StoreAction.teacher_slot = (teacher_index, K)`; no
  Top-M, no signals in the decision, no R, no P. `decide` takes
  `teacher_index=` / `num_teachers=` (the rank-synced representative
  teacher and the registry size, tag included).
- `FastStore.slice_bounds` / `compute_slice_masks`: contiguous rank slice
  of width `rank // K` per teacher (remainder unused; error if `rank < K`);
  `WriteExecutor._execute_f(..., teacher_slot=)` opens exactly that
  slice's A rows / B columns instead of Top-K.
- `MaskedAdamW.reset_state_for_coords(param, mask)`: masked counterpart of
  `reset_state_for_params` (moments, AMSGrad max, coordinate counter
  zeroed on the mask, untouched elsewhere; `step` unchanged; `step()` not
  modified).
- `DebugConfig.ablate_teacher_slice: {step, teacher}`: after the optimizer
  step at `step`, the teacher's slice is zeroed in every adapter with its
  optimizer state (an `ablation` event records `coords_zeroed`). A zeroed
  A row makes the slice's gradients vanish, so the region is dead — GR's
  "remove the region" — with no RNG and identical on every rank; the hook
  runs before the step's checkpoint so a resume never repeats it.
- `bad_teacher_small.yaml` gains `gradient_routing` (rank 16 / 5 teachers →
  3 components each; ablation at step 3649, the end of `math_recurrent`).
- Tests: `tests/test_gradient_routing.py` (+6).
- Deviation: the ablation zeroes rather than re-initialises the slice (no
  RNG, DDP-trivial, and the dead slice is the honest GR semantics).

## T9 — Part C reading notes, and the baseline fixes they forced

- `docs/related_work_notes.md`: the ten papers in the doc's fixed
  structure (Claim / Mechanism / Overlap / Our difference / Baseline
  implication / one sentence / BibTeX), read from the arXiv HTML/PDF on
  2026-09-04; `paper/refs.bib` (13 entries); `docs/related_work_draft.md`
  (four paragraphs, ~560 words, ending on the one-line gap).
- **MoLF's score is not m²/v.** Its EPD is `(η_i/N_i) Σ m²/(√v+ε)` per
  expert, an argmax between the dense and the LoRA expert of each module;
  `stability_adam` is (up to ε) the square of MoLF's PFN *ablation
  baseline*. Added `ModuleSignals.epd_score` (`StabilityTracker.
  compute_epd`, pre-update moments), `PolicyConfig.adam_score_rule:
  epd_argmax | snr_threshold` (default `epd_argmax`), `epd_lr_base /
  epd_lr_lora`, and `RFPPolicy._epd_argmax` (per block the P and F modules
  compete; the winner alone updates; every block routes every step; a
  blocked teacher's win goes to the LoRA expert). `molf_style` in
  `baselines_small.yaml` and `bad_teacher_small.yaml` now uses it with
  `writer.top_k_fraction=1.0` (MoLF's LoRA expert takes a full step); the
  SNR-threshold variants stay in the sweep as `molf_style_a*`, plus
  `molf_style_epd`. Not reproduced, on purpose: MoLF's universal momentum
  tracking (losers' moments advance on a shared clock) — Theorem 1.
- **Online-LoRA consolidates only on a plateau that follows a loss peak.**
  `ConsolidationConfig.plateau_require_peak` (default false; the E1/E3
  `plateau_trigger` specs set it true): a plateau fires only after the
  window mean rose by more than the window's std since the last fire;
  detector state round-trips. The author list in the task document ("Wei,
  Kim") is wrong — it is Wei, Li, Marculescu (WACV 2025, arXiv:2411.05663).
- **OGP (arXiv:2604.22407) was withdrawn by its authors on 24 Jul 2026.**
  Cited for its Proposition 1 only (the 1/(1−α) inflation argument that
  motivates exact masking); its numbers are not to be used.
- **STABLE (arXiv:2510.16089) has no baseline** — recorded as a follow-up
  in `docs/STATUS.md` with the design of a faithful `stable_gate`.
- Terminology for the paper: R is a *deferral* tier (not retrieval in the
  ReGrad sense); attribution is *provenance* (not importance as in
  Attribution-Guided CL); never "optimizer as memory" (Nested Learning).
- **Bug found by the real-data E5 smoke:** the periodic sweep can merge
  through its legacy no-flags path while the pre-merge checkpoint was
  gated on `pending_p` being non-empty — a merge with no rollback point.
  `ConsolidationScheduler.select()` now exposes the modules that are about
  to merge and the loop saves `pre_merge` whenever that list is non-empty
  (`tests/test_attribution.py::test_every_sweep_merge_has_a_pre_merge_checkpoint`).
- Tests: +3 (`test_baseline_policies.py` EPD argmax + peak precondition,
  `test_stability.py` EPD formula, `test_attribution.py` sweep checkpoint).

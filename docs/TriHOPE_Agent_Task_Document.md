# TriHOPE — Agent Task Document (ICLR 2027 push)

**Audience:** a Claude coding agent working in the `Council-TriHOPE-main` repository.
**Companion document:** `TriHOPE_ICLR2027_Reframe.md` (the vision and the "why"). This file is the "what" and the "how".
**Hard deadline:** paper Sep 25, 2026 AoE. Abstract Sep 18. Internal go/no-go on E1 results: Sep 13.

Read this whole file before writing any code. Tasks are ordered by dependency and by importance. Do them in order unless a task's "Blocked by" field says otherwise.

---

## Part A — Context you need before touching anything

### A.1 What the repository does (one screen)

Each training step:

1. Embed the batch → pick a teacher (cosine prototypes or dataset metadata).
2. Student forward + loss `λ_kd·KD + λ_ce·CE + λ_reg·reg`. In cache mode KD is inert (no logit NPZs are published); the objective is CE on `teacher_output_text`, optionally scaled by `teacher_confidence` (`distillation.ce_confidence_weighting`).
3. `backward()`.
4. **Before** `optimizer.step()`, `SignalComputer.compute_all` (`src/hivemind/controller/signals.py`) reads each module's gradient and Adam `m_{t-1}, v_{t-1}` from `MaskedAdamW.get_state_for_param` and produces per-module `ModuleSignals`: `surprise` (S), `stability_C` (instant cosine), `stability_C_sustained` (EMA of cosine, C̄), `stability_V` (volatility), `stability_adam` (m²/v), `repetition` (fused R with components mom/hash/ret), `grad_norm`.
5. `RFPPolicy.decide` (`src/hivemind/controller/policy.py`) selects Top-M modules by `grad_norm` and classifies each as `R`, `F`, or `P` in `_classify`. Order: P first (R ≥ repetition_medium ∧ C ≥ stability_high_C ∧ V ≤ stability_low_V), then R (S ≥ surprise_high ∧ R < repetition_low), else F.
6. `WriteExecutor.execute` (`src/hivemind/controller/writer.py`) turns actions into optimizer masks: R → nothing opened, entry appended to `RetrievalStore`; F → Top-K LoRA rank components opened; P → base or full adapter opened, F-type module flagged on `ConsolidationScheduler`.
7. `MaskedAdamW.step` with staged masks. Closed coordinates do not change at all (Theorem 1).
8. Every `consolidation.period` steps, `ConsolidationScheduler.consolidate` re-validates flagged modules against current signals and merges LoRA into base (`direct` or `distill`), then resets adapter + optimizer state.
9. Everything is logged: `events.jsonl` (one `decision` record per selected module per step with all signals, bucket, teacher; `consolidation` records; `phase_*` records), `ModuleLedger` (`src/hivemind/tracing.py`) in every checkpoint, `run_summary.json`.

The training loop is `src/hivemind/training.py::run_training_loop`. The relevant region is roughly lines 1090–1290 (signals → policy → writer → optimizer → consolidation → logging). Read it end to end once.

### A.2 The stream

`configs/stream_small.yaml` (Qwen3-0.6B, 6000 steps) and `configs/stream_headline.yaml` (Qwen3-1.7B, 12000 steps). Phase map for `stream_small`:

| phase | steps | mode | domain |
|---|---|---|---|
| general_warm | 0–499 | random | general |
| code_recurrent | 500–1999 | recurrent | code |
| novel_inject | 2000–2149 | novel | general |
| math_recurrent | 2150–3649 | recurrent | math |
| medical_recurrent | 3650–4949 | recurrent | medical |
| code_revisit | 4950–5399 | recurrent (reuse_from code_recurrent) | code |
| mixed_tail | 5400–5999 | mixed | all |

Schedule code: `src/hivemind/data/stream.py::StreamSchedule`. The schedule digest is stored in checkpoints and validated on resume; any change to data identity must change the digest (see `config_digest`).

Experiments are driven by manifests in `configs/experiments/*.yaml` executed by `scripts/run_experiment.py`. Analysis is `python -m analysis.run_report runs/<experiment>` producing tables/figures from `analysis/tables.py` and `analysis/figures.py`.

### A.3 Ground rules (non-negotiable)

1. **`pytest tests/ -q` stays green after every task.** Every new behavior gets a test in `tests/`. Tests are CPU-only, tiny dims, deterministic; external libs are monkeypatched.
2. **`tests/test_resume_exact.py` and `tests/test_determinism.py` must keep passing.** Anything you add to controller state must be included in `state_dict`/`load_state_dict` and in `CheckpointManager` (`src/hivemind/checkpoint.py`), or bit-exact resume breaks.
3. **Do not change `MaskedAdamW` semantics.** Theorem 1 tests (`tests/test_masked_adamw.py`) are the paper's formal claim. You may add read-only helpers; you may not alter `step`.
4. **Every new behavior is behind a config flag whose default reproduces current behavior.** Old configs must run unchanged. Add the flag to the relevant dataclass in `src/hivemind/controller/config.py` (or the data/train config), with a docstring that states the paper motivation.
5. **Distributed safety.** Any new per-step controller input that depends on batch content must go through `dist.sync_controller_inputs` (see `src/hivemind/distributed.py`) or be derived from already-synced values. Random decisions must use a seeded generator that is identical on every rank.
6. **Preflight before GPU.** `python -m hivemind preflight --config-name <cfg> --metadata-only` must pass for every new config/spec.
7. **Write down what you did.** Each task ends with an entry in `docs/CHANGELOG_iclr.md`: what changed, which files, which tests, which config keys, and any deviation from this document with the reason.
8. **Commit per task**, message `T<n>: <title>`.

### A.4 Definitions used below

- **Permanent write:** any coordinate of a base (P-type) parameter that was opened for an accepted optimizer step, plus any LoRA→base merge. Count from `ModuleLedger` and `consolidation` events.
- **Active fraction:** `write/coords_opened` divided by the number of controller-indexed coordinates.
- **Retention delta:** per-domain eval loss/EM at a later phase boundary minus its value at the end of that domain's own recurrent phase (see `evaluation.py` retention tracker, `retention_table`). Positive = forgetting.
- **Steps-to-recover:** on `code_revisit`, steps until eval loss on code returns to within 5% of its value at the end of `code_recurrent`.

---

## Part B — Coding tasks

### T0 — Environment and baseline sanity

**Goal:** know the repo runs before changing it.

1. `pip install -e ".[dev,data,analysis]"`.
2. `pytest tests/ -q`. Record the count.
3. `python train.py --config-name stream_smoke` (CPU, ~5 min). Confirm `logs/run_summary.json` exists, `events.jsonl` has `decision` records, routing goes `R=4` early then `F=4` (see `docs/experiments.md`, "Warm-up smoke").
4. `python train.py --config-name stream_smoke checkpoint.resume_from=50` — final loss must equal the uninterrupted run's final loss to the last digit.
5. `python -m hivemind preflight --config-name stream_small --metadata-only`.

**Acceptance:** all five pass. Write the numbers in `docs/CHANGELOG_iclr.md` under T0.

---

### T1 — Sustained stability everywhere, and make P fire

**Why:** the policy's P branch and the consolidation re-validation both read the *instantaneous* cosine `stability_C` by default. On real text that is noise at any single step, so P never fires and the consolidation study is a null result. The repo already tracks the EMA `stability_C_sustained` (C̄) in `StabilityTracker`; consolidation can already read it via `consolidation.stability_mode=sustained`, but the **policy** cannot. This is also the answer to the "cos(g,m) is noisy" critique reviewers will raise.

**Changes:**

1. `src/hivemind/controller/config.py::PolicyConfig` — add `stability_source: str = "instant"`, allowed `"instant" | "sustained"`. Validate in `__post_init__` like `ConsolidationConfig` does.
2. `src/hivemind/controller/policy.py::RFPPolicy._classify` — when `stability_source == "sustained"`, use `sig.stability_C_sustained` in place of `sig.stability_C` for the P test. Leave the R and F logic untouched.
3. Make sure `stability_C_sustained` is still set to the neutral value when `cosine` is ablated (it already is in `SignalComputer`; verify).
4. In `configs/stream_small.yaml` and `configs/stream_headline.yaml` set:
   ```yaml
   controller:
     policy:
       stability_source: sustained
     consolidation:
       stability_mode: sustained
       period: 250
       min_stability_C: 0.4
       min_repetition: 0.4
   ```
   Keep `stream_smoke` unchanged.
5. Add `configs/experiments/ablation_grid.yaml` entries: `stability_instant` (`controller.policy.stability_source=instant controller.consolidation.stability_mode=instant`) and `consolidation_strict` (`controller.consolidation.min_stability_C=0.7 controller.consolidation.min_repetition=0.6 controller.consolidation.period=1000`).

**Tests:** `tests/test_policy.py` — a module with low instant C but high C̄ is classified P under `sustained` and F under `instant`. Extend `tests/test_stream_configs.py` if it validates shipped configs' values.

**Acceptance:** tests green; `stream_smoke` still bit-exact on resume; `preflight` passes on both stream configs.

---

### T2 — R → F replay promotion (deferral, not deletion)

**Why:** `WriteExecutor._execute_r` appends a `RetrievalEntry` and returns. Nothing ever reads the R-store back except an eval hit-rate. So an R action is simply a dropped update, and TriHOPE's low forgetting is confounded with not learning the new domain. The paper needs R to be a *deferral*: an observation is parked, and when the same pattern recurs, the parked evidence is replayed into fast weights. This is the single most important correctness fix for the paper's headline claim.

**Design (keep it minimal and deterministic):**

1. Config, in `src/hivemind/controller/config.py`, new dataclass:
   ```python
   @dataclass
   class RetrievalConfig:
       replay_on_hit: bool = False        # default off → current behavior
       hit_threshold: int = 2             # replays fire once a bucket has this many R entries AND a new hit arrives
       replay_batches: int = 1            # how many stored samples to replay per trigger
       replay_target: str = "F"           # store the replayed update is routed to ("F" only for now)
   ```
   Add `retrieval: RetrievalConfig` to `ControllerConfig`; wire it in `training._build_controller_config`.

2. `src/hivemind/stores/retrieval.py::RetrievalStore` — add:
   - `entries_for_bucket(bucket_id) -> list[RetrievalEntry]`.
   - `pop_for_bucket(bucket_id, n) -> list[RetrievalEntry]` (FIFO order, removes them from the buffer so a replayed item is not replayed forever).
   - Store `input_ids` on `RetrievalEntry` (add field `input_ids: torch.Tensor | None = None`, CPU tensor). The writer already receives `teacher_output_text`; extend `WriteExecutor.execute` and `_execute_r` to accept and store `input_ids` and `labels` for the representative row. Storage cost is bounded by `buffer_size`.

3. Trigger point. In `training.py`, immediately after `writer.execute(...)` returns (Step 10) and before the optimizer step:
   - Ask `r_store.entries_for_bucket(bucket_id)`. If `replay_on_hit` is on, the current batch's bucket has ≥ `hit_threshold` stored entries, **and** the current step's action for at least one module was not R (i.e. the pattern has started recurring), then pop `replay_batches` entries and push them onto a new `replay_queue` (a `deque` owned by the loop).
   - At the **start of the next step**, if `replay_queue` is non-empty, the batch for that step is the replayed sample instead of the stream sample, and the policy is bypassed: every Top-M module gets an `F` action. Log the decision with `"replay": true` and the original R step in `events.jsonl`. Do **not** advance the stream schedule on a replay step; the stream sample that would have been consumed is consumed on the following step. Simplest implementation: keep a `pending_stream_batch` slot; if `replay_queue` is non-empty, run the replay batch and keep the stream batch for later.
   - This keeps the stream digest unchanged (replay does not alter the schedule) and keeps determinism (replay order is FIFO on a seeded stream).

4. Checkpointing: `replay_queue` and the extra `RetrievalEntry` fields must be saved/restored by `CheckpointManager`, otherwise resume is not bit-exact. Add them where the R-store is already serialized.

5. Distributed: `bucket_id` is already synced. `entries_for_bucket` is deterministic given identical R-stores on every rank (they are — R writes are decided from synced inputs). Add an assertion in `dist.assert_rank_consistent` payload: `(step, "replay", len(replay_queue))`.

6. Metrics: `write/replay_count` per step; `retrieval/replayed_total` in `run_summary.json`.

**Tests (`tests/test_replay.py`):**
- Two R writes to the same bucket, then a non-R step on that bucket → queue has one entry; next step is a replay step with all-F actions; stream sample is not lost (the step after replay consumes the deferred stream sample).
- `replay_on_hit=false` → queue always empty, behavior identical to before (compare losses to a run without the flag).
- Save at a step with a non-empty queue, load, continue → identical losses (extend `test_resume_exact.py` with `replay_on_hit=true`).

**Acceptance:** tests green; `stream_smoke` with `controller.retrieval.replay_on_hit=true` runs and shows `write/replay_count > 0` after the novel phase.

**Blocked by:** T0.

---

### T3 — Baseline controllers for E1

**Why:** the paper's central claim is that reading m/v beats simpler triggers. That is only shown if the simpler triggers are run on the same stream with the same student and budget. Four baselines are needed; three are small.

#### T3.1 `surprise_gate` (Titans-style) — config only

`controller.ablation.disable_signals=[repetition, cosine, volatility]`. With the existing neutral pinning, the P test passes on stability (neutral C and V) but `repetition` is pinned to `repetition_medium`, which makes the R branch fail (R < repetition_low is false). That is **not** a surprise-only gate. Fix: add `controller.ablation.neutral_repetition` support for a per-branch value, or simpler: add `PolicyConfig.mode: str = "rfp"` with a `"surprise_only"` mode in `_classify`:
```
if S >= surprise_high: return "R"   (or "F" if R is disabled)
return "F"
```
P is never chosen in this mode. Spec: `controller.policy.mode=surprise_only`.

#### T3.2 `molf_style` (two-tier Adam-score routing, no R, no merge)

Add `PolicyConfig.mode="adam_score"`. In `_classify`: no R branch, no P-by-signals; instead use `stability_adam` (m²/v, already computed) and a new threshold `PolicyConfig.adam_score_high: float = 0.5`. If `stability_adam ≥ adam_score_high` → `P` (open base weights for this block, but **do not flag for consolidation** — MoLF has no merge); else → `F`. Spec: `controller.policy.mode=adam_score controller.ablation.disable_stores=[R] controller.consolidation.period=0`. In `WriteExecutor._execute_p`, respect a new `WriterConfig.flag_p_for_consolidation: bool = True`; set it false in this spec so a P action on an F-type module just opens the adapter.

Note in the changelog that MoLF's exact scoring function (their "EPD score") should be read from the paper (Part C, paper 2) and matched as closely as the block-level indexing allows. If their score is not m²/v, implement theirs and document the difference.

#### T3.3 `plateau_trigger` (Online-LoRA-style)

Controller off for routing (`controller.enabled=false`, `train.trainable=lora`), all adapters always open, and a merge when training loss plateaus. Add to `ConsolidationConfig`:
```python
trigger: str = "signals"         # "signals" | "plateau"
plateau_window: int = 200
plateau_tolerance: float = 0.01  # relative improvement below which we call it a plateau
```
In `training.py` Step 12, if `trigger == "plateau"`, maintain an EMA of `loss/total`; when the relative improvement over `plateau_window` steps is below `plateau_tolerance`, merge **all** adapters (reuse `consolidator.force_consolidate()`), log a `consolidation` event with `"trigger": "plateau"`. Note that `consolidator` must still be constructed when `controller.enabled=false` for this path; check how `training.py` builds it and guard accordingly. Spec: `controller.enabled=false train.trainable=lora controller.consolidation.trigger=plateau`.

#### T3.4 `random_routing` (the control that proves signals carry information)

Add `DebugConfig.policy_override="random_matched"`. Semantics: at each step, for each selected Top-M module, sample the store label from a fixed categorical distribution instead of `_classify`. The distribution must match TriHOPE's **per-phase** action shares so that the budget is matched. Implementation:
- New config `DebugConfig.random_shares_path: str | None` pointing to an `action_share_by_phase.csv` produced by `analysis.run_report` on a completed `trihope` run (the file already has phase × store shares).
- On startup, load it; at each step, look up the current phase via `schedule.phase_at(step).name`, draw from `(p_R, p_F, p_P)` with a `torch.Generator` seeded from `(cfg.seed, step)` so every rank draws the same label.
- If the file is absent, fall back to uniform `(1/3, 1/3, 1/3)` and log a warning.
- Diagnostics fields on `StoreAction` are still populated from real signals so the trace stays informative.

**Tests (`tests/test_baseline_policies.py`):** each mode produces only the allowed labels; `adam_score` never flags consolidation; `random_matched` with a two-phase shares file draws labels whose empirical frequencies over 2000 draws are within 3 points of the file; plateau trigger fires on a synthetic flat loss series and not on a decreasing one.

**Manifest:** append to `configs/experiments/baselines_small.yaml`: `surprise_gate`, `molf_style`, `plateau_trigger`, `random_routing` (with `controller.debug.random_shares_path=runs/baselines_small_v1/analysis/action_share_by_phase.csv` — document that `trihope` must be run first). Set `seeds: [1337, 2024, 7]`.

**Blocked by:** T1 (thresholds), T2 (TriHOPE spec should run with `replay_on_hit=true`; add that override to the `trihope` spec).

---

### T4 — Corrupted-teacher stream

**Why:** E5 asks whether a bad teacher's influence stays out of permanent memory and whether it can be reverted. To measure that, one teacher must be made bad in a controlled, deterministic, digest-tracked way.

**Design:**

1. New data config block (`configs/stream_small.yaml`, under `data:`), default absent:
   ```yaml
   corrupt_teacher:
     enabled: false
     domain: math
     phase: math_recurrent      # only rows served during this phase are corrupted
     fraction: 0.5              # of rows in that phase
     mode: shuffle              # shuffle | degrade
     confidence: 0.2            # teacher_confidence assigned to corrupted rows
     tag: math_teacher_corrupted
   ```
2. Implement in `src/hivemind/data/stream.py` (not in the HF loader): `StreamSchedule._build` knows which global indices are served in which phase. After building, if corruption is enabled, select `fraction` of that phase's served indices with the phase RNG (`self._rng(phase_index, tag=99)`), and store the set as `self.corrupted_indices`. Include `(domain, phase, fraction, mode, confidence, tag)` in `config_digest` so a corrupted run can never resume from a clean checkpoint.
3. Apply the corruption at collate time: `DistillCollator` (`src/hivemind/data/collate.py`) receives rows; add an optional `corruption: CorruptionSpec` that, for rows whose global index is in `corrupted_indices`:
   - `shuffle`: replace `teacher_output_text` with the `teacher_output_text` of another corrupted row in the same batch (or a fixed-seed permutation across the phase if batch size is 1 — precompute the permutation in `StreamSchedule`).
   - `degrade`: truncate `teacher_output_text` to its first 25% of tokens and append a wrong final answer (for math, replace the last number with `number + 1`).
   - set `teacher_confidence = confidence`, set `teacher_id = tag` so attribution sees a distinct teacher.
   The global index must be available to the collator — check `_unpack_batch`/sampler; if only local rows are passed, add `global_index` to the row dict in `HivemindHFDataset.__getitem__`.
4. `teachers.teacher_ids` must include the corrupted tag mapped to the same domain so `MetadataRouter` does not fall back to index 0; `preflight` must accept it (extend `src/hivemind/data/preflight.py`).
5. Emit a `phase_start` event field `"corrupted_rows": n` and record `corrupted_indices` size in `run_summary.json`.

**Tests (`tests/test_corrupt_teacher.py`):** exact fraction corrupted; corruption only in the named phase; same seed → same corrupted set; digest changes when enabled; `shuffle` never maps a row to itself; `teacher_id` on corrupted rows equals the tag.

**Blocked by:** T0.

---

### T5 — Teacher attribution table and selective rollback

**Why:** the "attributable and revertible" pillar. The event trace already carries `teacher` on every `decision`, and `consolidation` events exist, but nothing joins them into "which teacher wrote how much where," and there is no way to revert only one teacher's promotions.

**T5.1 Attribution tables (`analysis/tables.py`):**

- `teacher_attribution(runs) -> DataFrame` with columns `run, seed, teacher, phase, R_count, F_count, P_count, coords_opened_F, coords_opened_P, consolidations_attributed`. A consolidation is attributed to the teachers of the `F` decisions on that module since its last merge (weighted by `coords_opened`). Read `events.jsonl` in order and keep a per-module accumulator.
- `containment(runs) -> DataFrame`: for the corrupted teacher only: share of its actions in R/F/P, coords it wrote to P, and consolidations attributed to it — per method.
- Add both to `analysis/run_report.py` output and a figure `containment_bars.png` in `analysis/figures.py`.

**T5.2 Ledger completeness check (`tests/test_ledger_completeness.py`):**
Run 20 steps on the tiny model; assert that the set of coordinates where `θ_T ≠ θ_0` (plus merged deltas) is exactly the union of coordinates recorded as opened in the ledger/events. This is the empirical form of the paper's new Corollary 3 and must be a test, not a claim.

**T5.3 Tag pre-merge checkpoints with attribution:**
In `training._pre_merge_save`, add to the checkpoint's `meta.json`: `pending_modules`, and for each, the attributed-teacher weights from the same accumulator as T5.1 (expose it from the loop as `attribution_state`). The accumulator must be part of the checkpointed state (bit-exact resume).

**T5.4 Selective rollback script `scripts/rollback_teacher.py`:**
```
python scripts/rollback_teacher.py --run runs/<exp>/<spec>-seed<s> --teacher math_teacher_corrupted --out runs/<exp>/<spec>-seed<s>-rollback
```
Algorithm (minimal, defensible version):
1. Read `events.jsonl`; find the first `consolidation` event whose attribution weight for `--teacher` is ≥ `--min-share` (default 0.5).
2. Restore the `pre_merge` checkpoint saved at that step.
3. Resume training from that checkpoint to the original final step with two overrides: `controller.ablation.disable_stores=[P]` **for modules whose pending attribution is dominated by the teacher** (add `DebugConfig.block_p_for_teachers: list[str]` to make this precise: a P action is demoted to F if the current batch's teacher is in the list), and `data.corrupt_teacher.enabled` unchanged (the bad data is still in the stream; we are testing containment, not filtering).
4. Evaluate at the end with the standard phase evals; write `rollback_summary.json` with recovery on each domain vs. (a) the original run and (b) a full restore to the pre-phase checkpoint that discards later learning (compute (b) by resuming from the checkpoint at the corrupted phase's start with the corrupted phase skipped — implement as `--baseline full_restore`).

**Tests:** `tests/test_rollback.py` on the tiny model — a forced merge attributed to teacher X, rollback restores the pre-merge base weights exactly, and subsequent P actions for X are demoted.

**Blocked by:** T4.

---

### T6 — Gradient-routing baseline for E5

**Why:** Cloud et al. (Gradient Routing, 2024) route each labeled data source's gradients to a reserved region and ablate that region to unlearn. A reviewer will ask why we don't just do that with teacher IDs. We must run it. Our differentiator is that we do **not** use the teacher label in the policy; GR does.

**Design:**
- New `PolicyConfig.mode="teacher_partition"`: LoRA rank `q` is split into `K` contiguous slices, one per teacher in `teachers.teacher_ids` (K = number of distinct teacher names, including the corrupted tag). At each step the batch's teacher index selects the slice; the writer opens exactly that slice's rank components (reuse `FastStore.compute_top_k_masks` with an explicit component list — add `compute_slice_masks(adapter, components)`), base weights stay closed, no consolidation. This needs `model.lora.rank ≥ K`; set `rank=16` for this spec.
- Ablation step at the end of the corrupted phase: `DebugConfig.ablate_teacher_slice_at: {step: int, teacher: str}` zeroes that teacher's slice of `A` and `B` and its optimizer state (`reset_state_for_params` on a mask — add a masked variant or zero the slice rows/cols directly and reset full state for simplicity; document which).
- Report containment (trivially 100% in F by construction), damage, and recovery after ablation, using the same E5 tables.

**Tests:** slice masks are disjoint across teachers and cover the rank; ablation zeroes only the named slice.

**Blocked by:** T4, T5.1.

---

### T7 — Analysis: budget curve and Pareto figure

**Why:** Figure 1 of the paper is forgetting vs. permanent writes, one curve per controller. Nothing produces it yet.

1. `analysis/tables.py::budget_curve(runs) -> DataFrame`: per run: `method, seed, threshold_tag, permanent_writes, active_fraction_mean, worst_retention_delta, mean_retention_delta, new_domain_final_loss (per recurrent domain), steps_to_recover_code`.
2. `analysis/figures.py::pareto_figure(df)`: x = permanent writes (log scale), y = worst retention delta; one marker per (method, threshold), mean ± std over seeds; second panel with new-domain loss on y. Save `pareto_budget.png`.
3. Threshold sweep manifest `configs/experiments/budget_sweep_small.yaml`: `trihope` at `surprise_high ∈ {1.0, 2.0, 4.0}` × `repetition_low ∈ {0.2, 0.3, 0.45}` (9 points), `surprise_gate` at 3 surprise thresholds, `plateau_trigger` at 3 tolerances, `molf_style` at 3 `adam_score_high` values. 1 seed for the sweep, 3 seeds for the chosen operating point.
4. `analysis/run_report.py` accepts multiple experiment dirs so E1 and the sweep can be plotted together.

**Blocked by:** T3.

---

### T8 — Manifests and run order

Create/update these manifests (all `base_config: stream_small`, `seeds: [1337, 2024, 7]` unless noted):

| file | specs | purpose |
|---|---|---|
| `baselines_small.yaml` | trihope (replay on), full_ft, lora_only, no_retrieval, no_consolidation, surprise_gate, molf_style, plateau_trigger, random_routing | E1 |
| `r_tier_small.yaml` | trihope_replay, trihope_r_terminal (`replay_on_hit=false`), fp_only (`disable_stores=[R]`), p_only (`policy.mode=adam_score` + always open) | E2 |
| `budget_sweep_small.yaml` | see T7 | E3 |
| `p_study_small.yaml` | existing, with T1 defaults | E4 |
| `bad_teacher_small.yaml` | trihope, full_ft, lora_only, molf_style, gradient_routing — all with `data.corrupt_teacher.enabled=true` and `controller.consolidation.checkpoint_before_merge=true`; then rollback for trihope and gradient_routing | E5 |

Run order and budget (single GPU each; parallelize with `--parallel-gpus N`):
1. `baselines_small.yaml --only trihope` (needed for `random_routing`'s shares file) → run `analysis.run_report`.
2. Rest of `baselines_small.yaml`.
3. `p_study_small.yaml` and `bad_teacher_small.yaml` in parallel with 2.
4. `r_tier_small.yaml`, `budget_sweep_small.yaml`.
5. Rollback scripts.
6. `python -m analysis.run_report` on everything; copy `analysis/` outputs to `paper/figures/`.

For every completed experiment write `runs/<exp>/RESULTS.md`: the table, the figure paths, and three sentences: what it shows, what it does not show, anything surprising.

---

## Part C — Reading tasks

Read each paper below and write the requested notes in `docs/related_work_notes.md` (one section per paper). Every section has the same structure:

- **Claim** (2 lines, in your words).
- **Mechanism** (what they compute, what they update, what they mask/merge — precise).
- **Overlap with TriHOPE** (be harsh; list every ingredient they already have).
- **Our difference** (only things we can *demonstrate*, not assert).
- **Baseline implication** (do we need to run it? if yes, which task above covers it).
- **One sentence for the related-work paragraph** and the BibTeX entry in `paper/refs.bib`.

Papers (fetch by arXiv id; read the method section fully, skim experiments for the setting and scale):

1. **MoLF — "Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation"**, arXiv:2605.07111. Extract the exact EPD score and the Top-K masked AdamW update; check whether they use a shared or per-expert step counter for bias correction (we use per-coordinate). Feed T3.2.
2. **OGP — "Hidden Failure Modes of Gradient Modification under Adam in Continual Learning, and Adaptive Decoupled Moment Routing as a Repair"**, arXiv:2604.22407. Extract their 1/(1−α) effective-LR inflation argument; check whether any of our F writes could trigger the same second-moment pathology (our masks freeze `v`, which is different — say why).
3. **Gradient Routing — Cloud et al.**, arXiv:2410.04332. Extract the ERA (Expand, Route, Ablate) recipe and the "absorption" effect. Feed T6. Note their unlearning baselines (data filtering, RMU) — we do not need RMU, but we must explain why data filtering is not available online.
4. **ReGrad — "Retrievable Gradients: Continual Post-Training Without Cumulative Weight Drift"**, arXiv:2606.15734. Extract the gradient bank + inference-time retrieval; we cite it as the inference-side complement of our deferral tier and must stop calling R "retrieval" in the sense they mean.
5. **Attribution-Guided Continual Learning for LLMs**, arXiv:2605.05285. Confirm their "attribution" is LRP importance, not provenance; write the disambiguating sentence.
6. **Titans**, arXiv:2501.00663, and **Nested Learning**, arXiv:2512.24695. Extract exactly how surprise is defined and used, and the fast/slow timescale structure. Write the "isn't your surprise just Titans'?" answer (theirs gates a learned memory module's writes; ours gates permanence of ordinary weights and uses `v` for scale normalization and `m` for direction).
7. **Online-LoRA / STABLE** — find the current arXiv ids by search (Online-LoRA: loss-plateau-triggered LoRA merge for task-free CL; STABLE: probe-gated). Extract the trigger definitions. Feed T3.3.
8. **Merge before Forget**, arXiv:2512.23017, and **Sparse memory finetuning (Lin et al.)**, arXiv:2510.15103 — one paragraph each; both are two-tier.
9. **Revisiting Data-Free KD with Poisoned Teachers (Hong et al., ICML 2023)**, arXiv:2306.02368 — cite as the evidence that untrusted teachers are a recognized threat; one paragraph.
10. **HippoRAG 2**, arXiv:2502.14802 — already cited; refresh the sentence to contrast "external forever" with "deferred then promoted."

After all ten: write `docs/related_work_draft.md` — a 500–600 word related-work section with the four paragraphs: (i) multi-teacher distillation decides who teaches; (ii) continual learning decides how much to change; (iii) memory substrates with different persistence (RAG/ReGrad, LoRA, sparse memory, merging); (iv) optimizer state as evidence (Adam, MoLF, OGP, Titans/NL) — ending with the one-line gap: *every rival is two-store and decides where once; we are three-store, decide when to promote, enforce it exactly, and can attribute and revert permanent writes.*

---

## Part D — Reporting back

When all tasks are done (or on Sep 13 for the go/no-go, whichever comes first), produce `docs/STATUS.md` with:

1. Task table: T0–T8, status, commit hash, test count.
2. E1 table and Pareto figure, with a one-line verdict against the pass condition in the reframe document §5 (E1) and §9.
3. E5 containment figure and rollback recovery, with the same verdict format.
4. Anything you were unable to do, with the specific blocker (missing data, OOM, ambiguous spec). Do not silently reduce scope.
5. A list of every place where you deviated from this document, and why.

If a task is ambiguous, pick the simplest interpretation that keeps determinism and the tests green, implement it, and record the choice in the changelog. Do not stop to ask unless the ambiguity would change an experimental conclusion.

# Campaign notes — ICLR-2027 run, 2026-09-12/13

Operator log for the cache-mode campaign. Feeds `docs/STATUS.md` §8 and each
`RESULTS.md`. Written during the run, not reconstructed after.

## 1. STOP-1 threshold decision

`configs/stream_small.yaml`, three lines (backup: `stream_small.yaml.pre_stop1.bak`):

| key | was | now |
|---|---|---|
| `policy.surprise_high` | 2.0 | **0.75** |
| `policy.stability_high_C` | 0.5 | **0.30** |
| `consolidation.min_stability_C` | 0.4 | **0.25** |
| `policy.repetition_low` | 0.3 | **0.3 (unchanged)** |

### Why

With the shipped thresholds, one full 6000-step run (3 seeds) gave:

- `novel_inject` R share **0.8 / 1.4 / 0.4 %** — the phase built to route to R did not.
  Cause: `novel_inject` surprise is p50 0.75 / p90 1.05, so the 2.0 gate was never cleared.
- **P decisions 0 / 1 / 1** and only 2–7 consolidation events.
  Cause: C̄ maxes at 0.33–0.42 in every recurrent phase, so the 0.5 gate was unreachable.
  `V` passed 94–100 % and `repetition` 5–26 %, so C̄ was the sole blocker.
  The 250-step sweep was blocked the same way (`min_stability_C` 0.4 > observed max).

`permanent_writes.total` was NOT zero (15.7M / 47.2M / 53.5M coords) — permanence
arrived via the sweep, not via P, exactly as the guide's §9 caveat describes.

### How the value was chosen

NOT by taking `novel_inject`'s median surprise — that is circular, since it
guarantees ~50 % R in the phase used to judge success. Instead, a cost curve over
all 3 seeds, conditional on `Rp < repetition_low` (only those samples can reach R):

| `surprise_high` | 2.00 | 1.00 | **0.75** | 0.50 |
|---|---|---|---|---|
| novel R recall | 0.9 % | 13.0 % | **44.3 %** | 89.4 % |
| worst recurrent leak | 14.2 % | 17.7 % | **18.6 %** | 19.8 % |
| margin | **−13.4** | −4.7 | **+25.7** | +69.5 |

The shipped 2.0 has a NEGATIVE margin: it leaked more into recurrent phases than it
captured in the phase designed for R. Max-margin is 0.50, but that routes ~90 % of
`novel_inject` to R (which writes nothing); 0.75 matches the guide's stated "around
half" target. Measured after the change: **47.6 / 48.2 / 53.9 %**.

### Finding worth reporting: surprise is near-uninformative in cache mode

Leakage is nearly FLAT (14.2 → 19.8 %) across the whole threshold range while recall
swings 0.9 → 89.4 %. `Rp < 0.3` alone already selects 99.3 % of `novel_inject` but
only 8.5–21.9 % of recurrent phases. Repetition does essentially all the
discriminating. Directly relevant to the `no_surprise` entry in `ablation_grid`.

### Evidence that 0.30 did not simply admit junk

`p_selection_stats` on the retuned run: **mean C̄ at P = 0.4992 / 0.4835 / 0.4956** —
the P decisions that fire still sit at the OLD 0.5 gate. The other conjuncts
(`Rp >= 0.5`, `V <= 0.3`) select high-C̄ events on their own; the 0.5 gate was
clipping a distribution sitting just beneath it.

### Limitations (state these in the paper)

1. Calibration and evaluation used the same stream; the cost curve mitigates but does
   not remove this. No held-out novelty split exists.
2. Counterfactual replay is **off-policy** — changed routing changes the trajectory,
   so replay cannot predict equilibrium. (It predicted 44.3 %, actual 47–54 %.)
3. **Two coupled gates were changed at once** (R gating and P gating), so their
   downstream effects cannot be attributed separately.
4. Live-KD mode reached 103 P decisions with the ORIGINAL thresholds. The thresholds
   are therefore mode-dependent, not universally wrong; cache-mode CE genuinely
   produces lower directional stability than KD against a fixed teacher.

## 2. Defects found in the codebase

| # | Defect | Status |
|---|---|---|
| 1 | Rank-0 eval handler caught every exception and printed only `repr(exc)`; the all-reduce then re-raised a fresh error, so the original failure site was unrecoverable even with `HYDRA_FULL_ERROR=1`. | **Fixed**, commit `ebfa6bd` — `traceback.print_exc()` at the catch site. Error path only, cannot affect numerics. |
| 2 | **Resume was broken on GPU.** A resumed run died at the first eval with a CPU/CUDA mismatch. Root cause: `checkpoint.py` loaded `stores.pt` with the caller's CUDA `map_location`, but the R-store is CPU by contract (`state_dict` writes `.cpu()`, `WriteExecutor` adds `.cpu()`). Restored entries landed on the device while new ones stayed on the host, so the first `torch.stack` over that mixed buffer in `RetrievalStore._aligned` failed. GPU-only — the CPU suite has both sides on one device. Found via the traceback added in defect 1. | **Fixed** — load `stores.pt` with `map_location="cpu"`, matching the three neighbouring loads that already did. |
| 3 | Checkpoints are **4.6 GB**, not the ~1.9 GB the guide states (`optimizer.pt` 2.43 + `base.pt` 1.50 + `controller.pt` 0.91). Makes the guide's "≥300 GB" insufficient. | Documented; worked around by pruning between stages. |
| 4 | E5 (`bad_teacher_small`) sets `checkpoint_before_merge=true` + `keep_tagged=0`, so each of 21 runs keeps EVERY pre-merge checkpoint. Measured 7–10 merging sweeps/run → 9–12 ckpts → 41–55 GB/run → **~1.0 TB**. The retune made this worse (consolidations 2–7 → 10–16). | Worked around: E5 batched by spec, pruned between batches. |
| 5 | `analysis/tables.py:teacher_attribution` uses `DataFrame.iterrows()` over ~1.08M decision events with a dict-valued `signals` column; a 30-run report takes ~40 min. Runs ~6× per campaign. | **Open** — deliberately NOT patched mid-campaign; this code generates the paper's tables. |

## 3. Environment notes

- GPU is an **RTX PRO 6000 Blackwell (sm_120, 102 GB)**, not an A6000. Needs cu128
  wheels; torch 2.11.0+cu128 works.
- **NVML is broken** (kernel module 595.84 vs userspace 595.91) so `nvidia-smi` fails,
  but the CUDA driver path is intact and training is unaffected. A reboot restores
  telemetry. `torch.cuda.*_memory_allocated` still works (allocator, not NVML).
- Card measures **334.5 TFLOPS bf16**; the workload uses ~7 % of it. The job is
  launch-bound, not compute-bound.
- **MPS matters**: without it, concurrent CUDA processes are driver time-sliced.
  Measured 3 concurrent: 62.3k → 71.9k launches/s (+15 %). Saturates at 3 processes;
  6 and 9 are slightly worse.

## 4. Impact of defect 2 on E5

The resume path is used by exactly one thing: `rollback_teacher.py`, which restores a
pre-merge checkpoint and replays forward. So the bug hit **only the rollback
counterfactuals**, never the primary training runs.

Verified from `events.jsonl`: every primary E5 run shows no `resume` event,
`decisions=36000`, `max_step=5999` — complete and clean. The four runs that did
resume are the rollback replays, and each died ~1500 steps after its restore point
(3499 / 3747 / 3749 / 3750) instead of running to 6000.

**Consequence:** E5's headline containment result is unaffected — `rollback_teacher.py`
reported "no consolidation event attributes >= 0.50 of a merge to
'math_teacher_corrupted'; nothing to roll back (containment held)" for the trihope
seeds, which is a result, not an error (guide 5). But the selective-rollback vs
full-restore *recovery curves* were truncated. `trihope`'s checkpoints were pruned
after its rollbacks ran, so regenerating those curves requires re-running the three
`trihope` E5 runs with the fix in place.

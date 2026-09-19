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

## 5. Campaign outcome (2026-09-14)

108 runs + 7 rollback replays + 5 follow-ups across six matrices. **Zero run
failures.** All reports written: `figure1`, `r_tier`, `p_study`, `ablations`,
plus per-matrix `analysis/`.

### The finding that decides the paper

E1 and E5 both showed large margins over `molf_style`. Both are artefacts of
one configuration choice: that baseline runs `adam_score_rule=epd_argmax`
**with `top_k_fraction=1.0`** (opens every coordinate) while `trihope` runs at
the default 0.5. It makes 9.3x more decisions and 26,000x more permanent
writes, so it forgets catastrophically and writes the corrupted teacher
straight to P.

`budget_sweep_small.yaml` already ships the fair variant
(`adam_score_rule=snr_threshold`, top-k inherited at 0.5). Run at 3 seeds:

| axis | `trihope` | `molf` (fair) | winner |
|---|---|---|---|
| worst retention delta | 0.1883 +/- 0.0036 | **0.0841 +/- 0.0062** | molf, 2.24x, non-overlapping |
| corrupted-teacher P share | 0.00% | **0.00%** | tie |
| coords direct to base | 0 | **0** | tie |
| permanent writes | 1.35e8 | **8.18e7** | molf, -40% |

**`trihope` does not win on any axis.** The fair MoLF is not trivially
containing — it writes 8.18e7 coords permanently, it simply never routes the
corrupted teacher there.

This is the **no-go** branch. Recording it here rather than in a reviewer's
report: the sweep that refutes the claim is in this repo.

### What survives

- **Signals carry information**: `trihope` 0.188 vs `random_routing` 0.650
  (same action budget, shuffled assignment), non-overlapping.
- **C-bar is the mechanism**: `no_cosine` -> 0.354 and 1.79e10 writes (200x).
- **Surprise and replay are harmful**: `no_surprise` 0.117, `r_terminal` 0.119,
  both better than `trihope` 0.188.
- **Timing of permanence matters**: forced merges cost ~28%, worst at low
  confidence.
- **The configuration, not the mechanism, is wrong**: `consolidation_strict`
  (0.1153 @ 8.81e7) dominates tuned `trihope` on both axes.

### Where the STOP-1 decision went wrong

The retune loosened the stability gates (`stability_high_C` 0.5 -> 0.30,
`min_stability_C` 0.4 -> 0.25) because P never fired. That optimised for
"make P fire" rather than for retention per write. `consolidation_strict`
goes the opposite way (0.7 / 0.6) and beats it on both. The original strict
direction was closer to right, and stricter still looks better.

## 6. Reported error bars understate the true uncertainty

`baselines_small_v1/trihope` and `r_tier_small_v1/trihope_replay` are the same
configuration. Their override files diff to nothing — both are
`++controller.retrieval.replay_on_hit=true ++train.seed=<s>`. Same code, same
machine, same seeds. They do not produce the same numbers:

| seed | E1 `trihope` | E2 `trihope_replay` | abs diff |
|---|---|---|---|
| 1337 | 0.1899 | 0.1999 | 0.0100 |
| 2024 | 0.1833 | 0.1783 | 0.0050 |
| 7 | 0.1917 | 0.2035 | 0.0119 |

Mean repeat-run difference **0.0089**, against a reported seed-to-seed sd of
**0.0036**. GPU nondeterminism (non-deterministic kernels, atomics, cuDNN
algorithm selection) is ~2.5x the variance the tables show, so every `+/-` in
this campaign **understates true uncertainty by roughly 2-3x**.

**Unaffected** (gaps 8-51x the noise): trihope vs `random_routing` (0.462),
`molf_snr` vs trihope (0.104), `no_cosine` vs trihope (0.166), `r_terminal`
vs `trihope_replay` (0.075).

**Not resolved** (inside the noise): trihope vs `no_consolidation` (0.0038),
`stability_instant` vs trihope (0.0031), `r_terminal` vs `fp_only` (0.0017),
`consolidation_strict` vs `no_surprise` (0.0016), and the `surprise_gate`
separation already flagged as convention-dependent.

The n=1 matrices (E3, E4, ablations) carry this +/-0.009 with no error bar
shown at all. Any ordering among them tighter than ~0.02 should not be
claimed.

**For the paper:** report repeat-run variance alongside seed variance, or
state plainly that differences below ~0.02 are unresolved. Two identical
configurations with different numbers are already sitting in this repo.

## 7. consolidation_strict at n=3 — the tuning, not the mechanism

Re-ran `consolidation_strict` at seeds 2024 and 7 (it was n=1, and repeat-run
noise is ~0.009 per §6). It holds:

| config | worst ret D | +/- | perm writes |
|---|---|---|---|
| `molf_snr` (fair baseline) | 0.0841 | 0.0062 | 1.03e8 |
| **`consolidation_strict`** | **0.1194** | 0.0044 | **9.96e7** |
| `trihope` (tuned) | 0.1883 | 0.0036 | 1.31e8 |

`consolidation_strict` (`min_stability_C=0.7`, `min_repetition=0.6`,
`period=1000`) beats tuned `trihope` by 37% on retention with 24% fewer
permanent writes. Non-overlapping (0.1238 vs 0.1847); the 0.069 gap is ~8x
repeat-run noise. It has the fewest permanent writes of anything tested,
including the fair MoLF baseline.

**The STOP-1 retune went the wrong way.** It loosened the stability gates
(`stability_high_C` 0.5 -> 0.30, `min_stability_C` 0.4 -> 0.25) to make P fire
at all — optimising for "P fires" rather than retention per write. The strict
direction recovers 37% of the deficit without touching the mechanism.

**Open direction.** The gap to `molf_snr` is now 0.035, roughly half of what
it was, closed by configuration alone. Nobody has swept the strict region:
`consolidation_strict` is a single point, and E3's sweep only covered
`surprise_high` x `repetition_low` at the *loose* stability setting. The space
between `min_stability_C` 0.25 and 0.7 — crossed with E3's best
`repetition_low=0.2` points (0.1302) — is unexplored and is where a
competitive configuration plausibly sits.

That is the difference between "the method loses" and "the method was
mis-tuned"; it needs a sweep, not a rewrite.

## 8. CORRECTION to §7 — the ablations were read against the wrong baseline

Upstream commit `b07a68f` flags a confound I missed: **`ablation_grid` and
`p_study_small` ran with replay OFF**, while the shipped method has replay ON.
Every ablation must therefore be read against the replay-off baseline
(`r_terminal` 0.119 / `no_retrieval` 0.119 / `fp_only` 0.121), **not** against
E1 `trihope` (0.188).

That invalidates two claims I made:

1. **"11 of 12 ablations beat trihope"** — wrong reference. Against the
   replay-off baseline the correct reading is upstream's: cosine is essential
   (+0.24 and ~100x the writes without it), sustained C-bar matters (+0.07),
   confidence and repetition help a little, **surprise and strict
   consolidation are neutral**.
2. **§7's "consolidation_strict dominates the tuned config"** — confounded.
   `consolidation_strict` never sets `replay_on_hit`, so it inherits the
   default (off). Its 0.1194 +/- 0.0044 at n=3 is *level with* the replay-off
   baseline (0.119-0.121), not above it. It appeared to win only because the
   comparison target carried replay, and replay is what costs 62%.

The n=3 repeat was still worth running — it is a clean measurement — but the
conclusion drawn from it was wrong. The correct statement is: **stricter
gating is neutral once replay is held constant; replay is the dominant
effect.** Whether strict gating helps the *shipped* method is untested until
the grid is rerun with replay on.

Which is exactly why the reruns matter: the grid and E4 must ablate the
method actually being proposed.

---

## §9 — the reruns landed; §8 is now resolved (2026-09-19)

The replay-on grid §8 called for is complete: `runs/ablation_grid_v1`, 13 arms,
every one `state=done, exit_code=0` at 6000 steps, and — the part that matters —
it carries its **own replay-on `trihope` arm**, so the deltas are finally
measured against a baseline run under identical settings. Full table in
`docs/E4_CORRECTED_ABLATION_2026-09-19.md`; consolidated view across E1/E4/E5 in
`docs/RESULTS_CONSOLIDATED_2026-09-19.md`.

Corrected baseline: in-grid replay-on `trihope` = **0.0961** mean forgetting.

What §8 got right, and what it still got wrong:

- **Right:** the original "11 of 12 beat trihope" was a baseline error. The
  corrected count is **4 of 12**.
- **Right:** `consolidation_strict` is neutral. At -0.0053 it sits at 0.9x the
  seed-noise floor — indistinguishable from baseline, exactly as §8 predicted
  once replay is held constant.
- **Wrong:** §8 recorded surprise as "neutral" (reading the replay-off grid).
  With replay on, `no_surprise` costs **+0.0936**, ~16x noise. Surprise is not
  neutral; it was masked by the replay-off confound in the other direction.
- **Confirmed and strengthened:** cosine is essential (`no_cosine` +0.1117, 19x),
  and raw Adam moments are worse still (`moments_optimizer` +0.1536, 26x).

The genuinely new finding the rerun exposes is that three ablations *beat* the
shipped configuration by margins well outside noise: `no_repetition` (-0.0448,
7.6x), `topm_2` (-0.0270, 4.6x), `topk_25` (-0.0207, 3.5x). Read together with
E1 (`no_retrieval` -0.0426 at n=3) and E5 (`trihope_nogate` ties to four
decimals), the pattern is consistent: the controller's *surprise and cosine*
signals carry the method, while repetition, retrieval, consolidation and the
confidence gate do not, and the shipped write budget is too large.

Noise floor used throughout: the only n=3 arm available (`consolidation_strict`,
replay-off) gives std ~0.0042 on the mean-forgetting statistic, so a
single-seed difference carries ~0.0059. E4 is n=1 per arm; deltas in the 2-4x
band need n=3 before they go in a paper. This borrows one arm's variance for
all arms and should be stated as such.

---

## §10 — STOP-1, live mode (2026-09-19)

The live pass hit the same gate §1 describes, for a different reason, and the
thresholds had to be re-derived rather than inherited.

**The box was running the shipped thresholds, not §1's.** `configs/stream_small.yaml`
on the GPU box was byte-identical to `stream_small.yaml.pre_stop1.bak` — the §1
retune (`surprise_high` 0.75, `stability_high_C` 0.30, `min_stability_C` 0.25) was
never persisted there. So the first live gate ran at `surprise_high: 2.0`, the
value §1 measured at a **negative** margin (−13.4).

Result: `novel_inject` R share **107/900 = 11.9%**, against the guide's "around
half" target. Live `novel_inject` surprise is p50 1.658 / p90 4.158, so the 2.0
gate is cleared by a minority — the same failure mode as cache mode (0.8/1.4/0.4%),
just less severe because live surprise runs higher.

**Why the values were not inherited from §1.** Live and cache differ in both
signals that matter here. Live `novel_inject` surprise p50 is 1.658 against cache's
0.75, and live repetition has shifted up (p10 0.227 / p50 0.373 / p90 0.409), so
`Rp < 0.3` — which selected 99.3% of `novel_inject` in cache mode — now selects
only ~12%. In cache mode surprise was the blocker; in live mode repetition is.

**Joint sweep**, 108,000 decisions across the three live seeds, scored as in §1
(recall in the phase built for R against the worst leak into a recurrent phase):

| `surprise_high` | `repetition_low` | novel recall | worst leak | margin |
|---:|---:|---:|---:|---:|
| 2.00 (shipped) | 0.30 | 11.9% | — | negative per §1 |
| 0.75 | 0.38 | 51.4% | 16.1% | +35.2 |
| 1.00 | 0.38 | 47.5% | 14.4% | +33.0 |
| **1.25** | **0.40** | **57.6%** | **14.9%** | **+42.7** |
| 1.50 | 0.42 | 55.9% | 15.1% | +40.7 |

Applied: `surprise_high: 1.25`, `repetition_low: 0.40` — max margin inside the
40–60% band. Two lines; backup `stream_small.yaml.pre_live_stop1.bak`.

**The C̄ gates were deliberately left alone.** §1 lowered them because C̄ maxed at
0.33–0.42 in cache mode and blocked P entirely (2–7 consolidation events). Live
does not have that problem: C̄ reaches 0.906 and the first live gate logged **153
consolidation events**. Changing them would fix a failure that is not occurring.

The three old-threshold live runs are kept as
`runs/_live_stop1_evidence_oldthresholds/`, mirroring the cache-mode evidence dir,
rather than deleted.

**Consequence for the tables:** live numbers are not comparable to the cache-mode
tables, and not only because of the thresholds — in cache mode `lambda_kd` is
inert without logit caches (`loss/kd = 0.0`, `kd_row_hit_frac = 0.0`), so every
cached result measured CE + regularization only. Live is the only configuration
where the distillation term is actually non-zero (`loss/kd = 4.199`).

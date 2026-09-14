# TriHOPE — Operator guide for the ICLR-2027 campaign

**Audience:** the person running the GPU campaign. This is the one document
to follow. Every command in it has been executed on a 96 GB card or on the
CPU dev box against real data; the numbers quoted are observed. Written
2026-09-09 against `main` at `c5ebaaf` or later; progress log §2b updated 2026-09-14.

**Deadlines:** abstract Sep 18, paper Sep 25 (AoE). The E1 go/no-go was
planned for Sep 13.

Companion documents (read only when this one points at them):
`docs/RUN_PIPELINE_2026-09-04.md` (health signals per phase, troubleshooting),
`docs/GPU_RUNBOOK.md` (tiers), `docs/experiments.md` (what each experiment
answers), `docs/data_provenance.md` (where the data came from),
`docs/TriHOPE_ICLR2027_Reframe.md` (what the paper claims, §9 = go/no-go),
`docs/STATUS.md` (result slots you fill), `docs/CHANGELOG_iclr.md` (every
change, dated).

---

## 0. What you are running, in one paragraph

A pretrained 0.6B student learns from a 6000-step stream of real data in
seven phases (general → code → isolated novel rows → math → medical → code
again → mixed). Every step produces one gradient. A controller reads that
gradient together with its own Adam-style running moments and decides, for
each of the six most active blocks of the model, where the update should
live: **R** (defer: write nothing, park the example), **F** (tentative:
open a few coordinates of the block's LoRA adapter), or **P** (permanent:
consolidate the adapter into the base weights). Exact masking guarantees a
closed coordinate does not move at all. The paper's claims are comparisons
between *controllers* on that identical stream: less forgetting per
permanent write than the rules people already use (E1, Figure 1), the
no-write tier earns its place (E2), timing of permanence matters (E4), and
a bad teacher's influence can be contained and rolled back (E5, Figure 4).
There is no leaderboard number. Distillation is only the source of the
gradient.

---

## 1. Ingredients (what every run uses)

### 1.1 Student

| | |
|---|---|
| Model | `Qwen/Qwen3-0.6B` from the Hugging Face hub, bf16, gradient checkpointing |
| Context | 512 tokens (prompt + teacher answer; longer answers are cut) |
| Batch | 2 sequences per step |
| Optimizer | MaskedAdamW, lr 1e-4, betas (0.9, 0.999), weight decay 0.01 |
| LoRA | rank 16, alpha 32, on q, k, v, o, up, gate, down in all 28 layers |
| Stores | **P** = base weights · **F** = the LoRA adapters · **R** = an external embedding store (CPU, 1000 entries) |
| Controller index | 112 modules = 28 layers × {attention, FFN} × {base, LoRA}. Embeddings, norms and the output head are outside the index and train normally |

The 1.7B profile (`stream_headline`: `Qwen/Qwen3-1.7B`, 1024 tokens, 12 000
steps, batch 1) is optional and comes last.

### 1.2 Data: four public datasets

All under `hivemind-research/` on the hub, downloaded on first use
(~2 GB total; set `HF_HOME` to a large disk first). Each has train /
validation / test = 160k / 20k / 20k rows and the same 18-column schema:
`input_text` (the question), `teacher_output_text` (the answer), `teacher_id`,
`teacher_confidence`, `bucket_id` (the repetition key), `target_text` +
`has_gold_label`, plus metadata.

| Domain | Repo | Answer text written by | Gold labels | Mean confidence |
|---|---|---|---|---|
| general | `general-layerC-200k` | DeepSeek-R1-Distill-Llama-70B, shipped with `glaiveai/reasoning-v1-20m` | 0 % | 0.82 |
| code | `code-layerC-200k` (64 shards, ~870 MB) | DeepSeek-R1, shipped with `nvidia/OpenCodeReasoning` | 100 % | 0.83 |
| math | `math-layerC-200K` | DeepSeek-R1-Distill-Qwen-1.5B, generated for this project | 2 % | 0.88 |
| medical | `medical-layerC-200k` | Qwen2.5-1.5B-Instruct, generated for this project | 78 % | 0.59 |

The stream uses a slice of each (40k-row cap per domain, 8 fixed buckets
per recurrent phase). Code keys buckets on `subdomain × difficulty × source`
because its published `bucket_id` has only 3 values. Medical is the only
domain whose confidence dips below the 0.3 policy gate. Provenance caveats
(what "confidence" means per domain, truncation, unverified generation
logs): `docs/data_provenance.md`.

### 1.3 Teachers: two modes

**Cache mode — every experiment in the paper.** *No teacher model is
loaded.* The student trains with cross-entropy on `teacher_output_text`,
weighted per row by `teacher_confidence`. This is sequence-level
distillation. The KD term in the loss is inert (`loss/kd` = 0.0) because no
logit files exist. The four "teachers" are name tags used for routing and
provenance; `teachers.teacher_ids` in the config must equal the dataset's
`teacher_id` values, and a `router/miss_rate` above 0 at step 0 means they
do not.

**Live mode — Tier 4 only (done).** Four frozen Qwen-vocabulary models
score the same cached text under teacher forcing and the student matches
their full distributions (τ² · KL at τ = 4, plus CE at half weight):
`Qwen/Qwen3-4B` (general), `Qwen/Qwen2.5-Coder-1.5B-Instruct` (code),
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (math),
`Qwen/Qwen2.5-1.5B-Instruct` (medical); ~18 GB bf16 on top of the run.
General and code live teachers are *not* the models that wrote the cached
text (vocabulary mismatch), so live results are an appendix robustness
check, never part of Figure 1.

### 1.4 The stream (`configs/stream_small.yaml`)

| Phase | Steps | Domain | Mode | What it tests |
|---|---|---|---|---|
| general_warm | 0–499 | general | random | settle the optimizer statistics |
| code_recurrent | 500–1999 | code | 8 fixed buckets revisited every 10 steps | repetition and stability rise → F, then P |
| novel_inject | 2000–2149 | general | single-use rows, unique bucket ids | should route to R; **zero replays allowed here** |
| math_recurrent | 2150–3649 | math | recurrent | forgetting of code |
| medical_recurrent | 3650–4949 | medical | recurrent | confidence gate; more forgetting of code |
| code_revisit | 4950–5399 | code | the *same* buckets as code_recurrent | steps-to-recover (eval every 50 steps here) |
| mixed_tail | 5400–5999 | all four | interleaved | replay tail |

Eval: validation loss on all four domains every 250 steps and at every
phase boundary (16 batches per domain), plus exact-match generation on 64
samples for code, medical and math — these are the minutes-long pauses at
steps 499, 1999, 2149, … Checkpoint every 1000 steps; resume is bit-exact.

### 1.5 Controller defaults (`controller:` in `stream_small.yaml`)

| Setting | Value | Meaning |
|---|---|---|
| `policy.top_m_modules` | 6 | blocks decided per step, by gradient norm |
| `policy.surprise_high` | 2.0 | R needs surprise ≥ this … |
| `policy.repetition_low` | 0.3 | … and repetition < this |
| `policy.repetition_medium` / `stability_high_C` / `stability_low_V` | 0.5 / 0.5 / 0.3 | P needs repetition ≥, sustained cosine C̄ ≥, volatility ≤ |
| `policy.stability_source` | sustained | P reads the EMA C̄, not the instantaneous cosine |
| `consolidation.period` / `min_stability_C` / `min_repetition` | 250 / 0.4 / 0.4 | every 250 steps, F modules passing both are merged into base (P-flagged ones first; with none flagged, every F module is re-validated) |
| `writer.top_k_fraction` | 0.5 | share of a LoRA adapter's rank components an F write opens |
| `retrieval.replay_on_hit` / `hit_threshold` | true (trihope specs) / 2 | a parked bucket seen on 2 distinct earlier steps is replayed into F |
| `moments.source` / `sketch_stride` | tracked / 8 | the controller's own bias-corrected Adam-style m/v for every indexed coordinate (every 8th tracked; ≈ 0.6 GB). **Do not switch to `optimizer`** except through the `moments_optimizer` ablation: that setting reads MaskedAdamW's exactly-masked state, which is zero forever on never-opened base modules, so P can never fire |

The signals, in one line each: **surprise** = mean g²/v̂ (how unusual this
gradient is relative to history); **repetition** = 0.5 × bucket frequency +
0.3 × momentum agreement + 0.2 × retrieval-hit rate; **stability C̄** = EMA
of cos(g, m̂); **volatility** = windowed variance of the gradient norm.

---

## 2. What is already done

- **Tier 4, live logit-KD** (`runs/live_kd_small_v1`, five specs, seed
  1337) is complete on a 96 GB card. Its `run_report` tables go into
  `docs/STATUS.md` §4b. `trihope_live`: R 756 / F 35 141 / P 103, 138
  consolidations, 83 replays (0 in novel_inject), worst retention delta
  0.15, code recovers in 50 steps, 29 GB peak, 0.85 h.
- Every defect found on the first GPU days is fixed and dated in
  `docs/CHANGELOG_iclr.md` (2026-09-05 … 09-08): dead Adam signals →
  tracked moments; live-KD mask; manifest overrides forced to `++`;
  `surprise_gate` consolidating through the sweep; `--skip` for the runner;
  `scripts/diagnose_p.py`.

**Not done: the entire main campaign.** Every cache-mode run made before
`5788dac` is invalid and was deleted. Figure 1, Table 1, Figure 4, E2, E3,
E4 and the ablations do not exist yet. That is what you are producing.

---

## 2b. Progress log (update this as stages finish)

| Date | Stage | State |
|---|---|---|
| 2026-09-08 | Tier 4 live KD | done, 5 specs, reported (STATUS §4b) |
| 2026-09-14 | gate + e1 (E1, 33 runs) | done — **go** on the forgetting axis (§5b) |
| 2026-09-14 | tier3: E2 (15 runs), E3 sweep (22 runs) | done |
| — | e5 (21 runs + 9 rollbacks) | **next** |
| — | tier3: E4 `p_study_small` (5), `ablation_grid` (12), reports | after e5 |
| — | plasticity columns of E1/E2/E3, threshold-tag check, determinism check | owed (§5b) |

## 3. Machine and setup

- One 96 GB GPU is enough. Two cards halve the wall clock (`--parallel-gpus 2`
  in place of `--concurrent N`).
- Disk: **≥ 300 GB free**. E5 keeps every tagged checkpoint (~1.9 GB each)
  across 21 runs.
- Python 3.10+; the Python dev headers are required (Triton compiles a
  helper on the first forward).

```bash
git clone https://github.com/Hivemind-Home/Council-TriHOPE.git && cd Council-TriHOPE
git log --oneline -1                                   # c5ebaaf or later
sudo apt-get install -y python3.12-dev build-essential # use your interpreter's minor version
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,analysis]"                  # if bitsandbytes/kernels fail: pip install -e ".[dev,analysis]" datasets transformers huggingface_hub pyarrow accelerate
export HF_HOME=/path/to/big/disk/hf                    # put it in ~/.bashrc
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True
```

Observed footprint per run (0.6B, bf16, batch 2): cache mode ≈ 13 GB
(`full_ft` higher: full Adam state on every base weight), live mode ≈ 30 GB.
A lone cache-mode run does ~4 it/s; N concurrent runs share compute, so
the matrix's wall clock ≈ sequential hours ÷ 1 at best — concurrency fills
the card, it does not multiply throughput. `--concurrent 5` is the default
in the campaign script.

Run everything inside `tmux`. The runner writes each run's output to
`runs/<experiment>/<spec>-seed<seed>/stdout.log`, not to the terminal.

---

## 4. Sanity checks before any GPU time (≈ 15 min)

```bash
python -m pytest tests/ -q                                  # 514 tests; the 2 GPU-only ones run for real here
ruff check src tests analysis scripts                       # All checks passed!
python -m hivemind preflight --config-name stream_small --metadata-only   # "preflight OK"; validates teacher ids, buckets, phases
for m in baselines_small bad_teacher_small r_tier_small budget_sweep_small p_study_small ablation_grid; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --dry-run | tail -n 2; done   # prints each matrix (33/21/15/22/5/12 runs)
```

CPU smoke on real data (3–5 min, pinned to CPU by design; the numbers are
reference values from 2026-09-05 with tracked moments):

```bash
python train.py --config-name stream_smoke ++run.dir=runs/smoke/plain ++checkpoint.dir=runs/smoke/plain/ckpt \
    ++logging.path=runs/smoke/plain/metrics.jsonl ++logging.events_path=runs/smoke/plain/events.jsonl
#   Training complete. Final loss: 4.962428569793701      (actions R 62 / F 203 / P 15, 2 consolidations)
python train.py --config-name stream_smoke ++controller.retrieval.replay_on_hit=true ++run.dir=runs/smoke/replay \
    ++checkpoint.dir=runs/smoke/replay/ckpt ++logging.path=runs/smoke/replay/metrics.jsonl ++logging.events_path=runs/smoke/replay/events.jsonl
#   Training complete. Final loss: 4.934509754180908      (5 rows replayed)
python train.py --config-name stream_smoke ++controller.retrieval.replay_on_hit=true ++checkpoint.resume_from=50 ++run.dir=runs/smoke/resume \
    ++checkpoint.dir=runs/smoke/replay/ckpt ++logging.path=runs/smoke/resume/metrics.jsonl ++logging.events_path=runs/smoke/resume/events.jsonl
#   must print the SAME final loss as the replay run, to the last digit
```

If the plain smoke ends at `5.054330348968506`, the checkout predates the
tracked-moments fix: stop and update.

---

## 5. The campaign: four commands, two stops

`scripts/run_campaign.sh` runs the whole thing. Every stage is
**idempotent**: finished runs are skipped, interrupted ones resume
bit-exactly from their last checkpoint, existing rollback outputs are not
regenerated — so rerunning a stage after a crash or a Ctrl+C is always the
right move. `CONCURRENT=5` by default; `DRY_RUN=1` prints matrices only.

| Stage | Command | Runs | GPU-hours (sequential) | Ends with |
|---|---|---|---|---|
| gate | `scripts/run_campaign.sh gate` | trihope × 3 seeds | ~4 | report + signal diagnostic, **STOP 1** |
| e1 | `scripts/run_campaign.sh e1` | the other 10 E1 specs (`random_routing` last) | ~40 | Figure 1 inputs, **STOP 2** |
| e5 | `scripts/run_campaign.sh e5` | 21 corrupted-teacher runs + 9 rollbacks | ~30 | containment + rollback tables |
| tier3 | `scripts/run_campaign.sh tier3` | E2 (15), E3 (22), E4 (5), ablations (12) | ~75 | `runs/figure1`, E2/E4/ablation reports |

`scripts/run_campaign.sh all` chains the four without stopping — only for a
rerun after both decisions below have been made.

### STOP 1 — after `gate`: the threshold decision

The stage prints `scripts/diagnose_p.py` for `trihope-seed1337`: per phase,
the distribution of every signal and the share of decisions passing each
P condition. Two things to check.

1. **Signals alive.** C̄ spread over (0, 1) from `general_warm` on, surprise
   mostly below 20, a non-zero P count by the end of `code_recurrent`. The
   script aborts by itself if C̄ is `0.000` everywhere (stale checkout).
2. **The R tier is reachable.** Read the `novel_inject` block. It is the
   phase built to route to R. In the live run only 12 % did, because a
   pretrained student's surprise rarely clears 2.0 and the momentum term of
   repetition carries over between phases so novel rows rarely fall below
   0.3. If the cache-mode run shows the same (R share well under half in
   `novel_inject`), set in `configs/stream_small.yaml`:
   `controller.policy.surprise_high` ≈ the novel-phase surprise p50 (e.g.
   1.5) and `controller.policy.repetition_low` ≈ 0.4; then
   `rm -rf runs/baselines_small_v1` and rerun `gate`. **Decide this before
   `e1`: every baseline inherits these values, and the E3 sweep tags its
   points with them.** Record the decision in `docs/STATUS.md` §8.

### STOP 2 — after `e1`: the go/no-go (reframe §9)

Read `runs/baselines_small_v1/analysis/pareto_budget.png` and
`budget_curve.csv` (x = permanent writes as coordinate-steps on base
weights; y = worst retention delta, and mean final eval loss).

- **Go:** `trihope` (or `trihope_no_hash`) on or above the Pareto front of
  `molf_style`, `plateau_trigger`, `surprise_gate` at matched permanent
  writes, and separated from `random_routing` with non-overlapping error
  bars on worst retention delta.
- **Conditional go:** E1 is a tie but E5 shows containment and working
  selective rollback → attribution/rollback becomes the headline.
- **No-go:** a baseline dominates on E1 *and* E5 shows the corrupted
  teacher reaching P like the baselines → workshop + ICML (Jan 16/22, 2027).

Write `runs/baselines_small_v1/RESULTS.md` and fill STATUS §2 either way.
Note which baseline sat closest to trihope — E5 rollbacks and the optional
headline use it.

### 5b. What E1 / E2 / E3 said (operator report of 2026-09-14, worst retention delta, 3 seeds unless noted)

| Spec | Exp | Worst Δ | ± | Permanent writes | Reading |
|---|---|---|---|---|---|
| trihope | E1 | 0.188 | 0.004 | 1.3e8 | **the method; fewest writes of any writer** |
| trihope_no_hash | E1 | 0.232 | 0.050 | 4.0e8 | bucket counter earns its place |
| no_consolidation | E1 | 0.192 | 0.006 | 0 | P off: same forgetting, zero writes |
| no_retrieval | E1 | 0.119 | 0.009 | 2.1e8 | R off: *less* forgetting |
| trihope_r_terminal / fp_only | E2 | 0.119 / 0.121 | 0.010 / 0.003 | 2.0e8 | R never replayed / no R: *less* forgetting |
| trihope_replay | E2 | 0.194 | 0.011 | 1.4e8 | replay on: +62 % forgetting vs r_terminal |
| surprise_gate / plateau_trigger / lora_only | E1 | 0.35 / 0.36 / 0.37 | 0.15 / 0.11 / 0.11 | 0 | the trigger baselines: ~2× worse |
| random_routing | E1 | 0.650 | 0.103 | 2.1e8 | **the control: 3.5× worse — the signals carry information** |
| molf_style / p_only / full_ft | E1/E2/E1 | 0.81 / 0.91 / 5.1 | 0.14 / 0.36 / 2.8 | 2e12 / 2.6e11 / 2.6e12 | writing to base freely forgets catastrophically |
| best sweep point trihope_s4_r0p2 (1 seed) | E3 | 0.130 | — | 1.7e8 | fewer R decisions → less forgetting; the s1_r0p45 point (most R) is worst at 0.238 |

**Verdict on reframe §9: go.** TriHOPE is on the Pareto front (nothing with
fewer writes forgets less; every external controller forgets ≥ 2× more) and
is separated from random_routing by far more than the error bars.

**What the ablations add, and the paper must say.** On the forgetting axis
the gain comes from signal-driven, sparse, exactly-masked F writes.
Replay (the R tier's return path) *increases* worst-case forgetting, and P
adds nothing to it. Neither was designed to reduce forgetting: replay
exists so deferred rows are still learned (plasticity, anytime loss), P so
adapters are reused and knowledge becomes base (E4). Their justification
therefore has to come from the columns not in the table above and from E5.
Until those are in, E2's claim is "deferral costs X forgetting and buys Y",
not "the no-write tier reduces forgetting".

**Owed before writing (operator, please send):**
1. From the same `budget_curve.csv` files: `final_loss_*`,
   `mean_retention_delta`, `steps_to_recover_code`, and
   `runs/figure1/pareto_budget.png` (both panels).
2. The `threshold_tag` of the E1 `trihope` rows — the sweep note says the
   shipped default is now surprise 1.0 (was 2.0). Figure 1 must state the
   values E1 ran with, and STATUS §8 must record the change.
3. Why `trihope` (E1, 0.188) and `trihope_replay` (E2, 0.194) differ: they
   are the same spec and seeds. Either the default changed between the two
   launches (then E2 was run against a different default — say which) or
   the GPU path is nondeterministic (then the error bars already cover it).
   Check: `diff <(grep -v hydra runs/baselines_small_v1/trihope-seed1337/hydra/rank0/.hydra/config.yaml) <(grep -v hydra runs/r_tier_small_v1/trihope_replay-seed1337/hydra/rank0/.hydra/config.yaml)`.

### What each stage executes (the manual equivalents)

```bash
# gate
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --only trihope --resume
python -m analysis.run_report runs/baselines_small_v1
python scripts/diagnose_p.py runs/baselines_small_v1/trihope-seed1337
# e1  (random_routing reads trihope's action-share file, hence the order)
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --resume --skip random_routing --concurrent 5
python -m analysis.run_report runs/baselines_small_v1
python scripts/run_experiment.py configs/experiments/baselines_small.yaml --resume --only random_routing --concurrent 3
python -m analysis.run_report runs/baselines_small_v1
# e5
python scripts/run_experiment.py configs/experiments/bad_teacher_small.yaml --resume --concurrent 5
for seed in 1337 2024 7; do r=runs/bad_teacher_small_v1/trihope-seed$seed
  python scripts/rollback_teacher.py --run $r --teacher math_teacher_corrupted --out $r-rollback
  python scripts/rollback_teacher.py --run $r --teacher math_teacher_corrupted --baseline full_restore --out $r-fullrestore
  g=runs/bad_teacher_small_v1/gradient_routing-seed$seed
  python scripts/rollback_teacher.py --run $g --teacher math_teacher_corrupted --baseline full_restore --out $g-fullrestore; done
python -m analysis.run_report runs/bad_teacher_small_v1
# tier3
for m in r_tier_small budget_sweep_small p_study_small ablation_grid; do
  python scripts/run_experiment.py configs/experiments/$m.yaml --resume --concurrent 5; done
python -m analysis.run_report runs/baselines_small_v1 runs/budget_sweep_small_v1 --out runs/figure1     # Figure 1
python -m analysis.run_report runs/r_tier_small_v1
python -m analysis.run_report runs/p_study_small_v1
python -m analysis.run_report runs/baselines_small_v1 runs/ablation_grid_v1 --out runs/ablations        # deltas vs trihope
```

`rollback_teacher.py` exiting 0 with "nothing to roll back (containment
held)" is a result, not an error; `--min-share 0.3` rolls back a partially
attributed merge.

---

## 6. What is in each manifest and why

**E1 `baselines_small.yaml` → `runs/baselines_small_v1`, 11 specs × 3 seeds.**
Same stream, same student, same LoRA budget; only the write policy varies.

| Spec | What it is |
|---|---|
| `trihope` | the method: R/F/P from the tracked moments, replay on |
| `trihope_no_hash` | same without the bucket-id counter (recurrence from optimizer + retrieval evidence only); the metadata-oracle control |
| `full_ft` | controller off, no LoRA, every base weight trains — writes everything, forgets most |
| `lora_only` | controller off, base frozen, every adapter always open |
| `no_retrieval` / `no_consolidation` | R branch off (falls to F) / P branch and sweep off |
| `surprise_gate` | Titans-style: surprise alone decides R vs F; never P, no sweep |
| `molf_style` | MoLF-style two-tier: base vs LoRA expert compete on an Adam-moment score; no R, no merge |
| `plateau_trigger` | Online-LoRA-style: adapters always open; merge when the loss plateaus after a peak |
| `random_routing` | trihope's per-phase action shares, assignment shuffled — proves the signals carry information |
| `gold_ce` | supervise on gold answers where they exist instead of teacher text (objective ablation) |

**E5 `bad_teacher_small.yaml` → `runs/bad_teacher_small_v1`, 7 specs × 3 seeds.**
Half the math buckets in `math_recurrent` get shuffled answers under the
tag `math_teacher_corrupted`; every pre-merge checkpoint is kept.
`trihope`, `trihope_nogate` (confidence gate off), `trihope_lowconf`
(corrupted rows also carry confidence 0.2), `full_ft`, `lora_only`,
`molf_style`, `gradient_routing` (Cloud et al.: one rank slice per teacher,
the corrupted slice zeroed at step 3649). Reports: containment (where the
bad teacher's actions went), damage on the other domains, selective
rollback vs full restore.

**E2 `r_tier_small.yaml` → `runs/r_tier_small_v1`, 5 × 3.** `trihope_replay`,
`trihope_r_terminal` (R writes nothing and is never read back),
`trihope_no_hash_replay`, `fp_only`, `p_only`. Key number:
`steps_to_recover_code` in `budget_curve.csv`.

**E3 `budget_sweep_small.yaml` → `runs/budget_sweep_small_v1`, 22 × 1.** Each
controller swept over its own threshold (trihope: surprise {1, 2, 4} ×
repetition_low {0.2, 0.3, 0.45}; surprise_gate S {1, 2, 4};
plateau_trigger tolerance {0.005, 0.01, 0.02}; molf EPD + SNR {0.3, 0.5,
0.7}). Pooled with E1 into `runs/figure1/pareto_budget.png` — the paper's
identity figure.

**E4 `p_study_small.yaml` → `runs/p_study_small_v1`, 5 × 1.** `p_study`,
`p_off_control`, and three forced merges at wrong times: `p_forced_bad`
(step 2075, mid novel_inject), `p_forced_plausible` (2350),
`p_forced_lowconf` (3850, medical). Reports `p_selection_stats` (when P
fires and the signals at those moments) and `damage_recovery`.

**Ablations `ablation_grid.yaml` → `runs/ablation_grid_v1`, 12 × 1.** One
signal removed per run (`no_surprise`, `no_repetition`, `no_cosine`,
`no_volatility`, `no_teacher_conf`), top-M {2, all}, top-K {25 %, 100 %},
`stability_instant`, `consolidation_strict`, and `moments_optimizer` (the
dead-signal behaviour, as a control). Reported against E1's trihope.

**Tier 4 `live_kd_small.yaml` — done.** **Headline `headline.yaml` —
optional last:** `trihope`, `trihope_no_hash` at 1.7B, plus the closest E1
baseline pasted in; ~6 h per run.

---

## 7. Monitoring and troubleshooting

```bash
tail -f runs/baselines_small_v1/trihope-seed1337/stdout.log             # one run, live
for f in runs/baselines_small_v1/*/status.json; do echo "$f: $(python -c "import json;print(json.load(open('$f'))['state'])")"; done
python scripts/diagnose_p.py runs/<experiment>/<run>                     # signal health, any time after ~step 500
```

Healthy `trihope` per phase (`RUN_PIPELINE` §4.1): miss rate 0 at step 0;
mostly F through `general_warm` (the pretrained student is not surprised by
general text); replays start after ~step 600 and P decisions plus
`consolidation` events appear at the 250-step sweeps in `code_recurrent`;
**zero replays in `novel_inject`**; code loss rises through math/medical
and recovers within a few hundred steps of `code_revisit`; the run ends
with `run_summary.json` present, `permanent_writes.total > 0`,
`retrieval.replayed_total > 0`, 7 retention entries.

| Symptom | Meaning / fix |
|---|---|
| runner prints the launch line, then nothing | normal — output is in `<run>/stdout.log`; `status.json` says `running` |
| `fatal error: Python.h` on the first forward | install `python3.X-dev`, relaunch with `--resume` |
| bar sits at step 499 / 1999 / 2149 / … for minutes | phase-boundary eval + exact-match generation; also every 250 steps and every 50 in `code_revisit` |
| `Key 'X' is not in struct` | only possible by hand now: use `++key=value` (manifest overrides get `++` automatically) |
| `router/miss_rate > 0` at step 0 | `teachers.teacher_ids` ≠ the dataset's values |
| `diagnose_p.py`: C̄ `0.000` everywhere | checkout predates `5788dac`; delete the run, update, rerun |
| a live run with `loss/kd` = 0 | predates `f9b422d`; delete, rerun |
| `surprise_gate` with consolidations > 0 | predates `371fad0`; delete, rerun |
| `permanent_writes.total == 0` at the end | P never fired — the diagnostic says which conjunct blocks |
| `random_routing` warns "shares file not found" | it ran before trihope's report; delete it and rerun after `run_report` |
| OOM | one run dies, the matrix continues; lower `CONCURRENT`, rerun the stage |
| `Stream config changed since the checkpoint was written` | you changed data/stream/corruption config and asked to resume: it is a fresh run |
| replicas "diverged across ranks" (DDP only) | file a bug; never lower `distributed.assert_rank_consistency` |

**Do not:** change any threshold after `e1` has launched; use DeepSpeed or
FSDP (rejected at startup — they break exact masking); run the headline
before Tier 3; delete `runs/live_kd_small_v1`.

---

## 8. Deliverables

After every `run_report`:

1. `runs/<experiment>/RESULTS.md` — the table, the figure paths, and three
   sentences: what it shows, what it does not show, anything surprising.
2. The matching slot in `docs/STATUS.md` (§2 E1, §3 E5, §4 E2/E3/E4, §4b
   Tier 4) with the verdict.
3. Commit `runs/**/analysis/*.csv`, `*.png`, `report.md` and every
   `RESULTS.md` (not the checkpoints), and push.

Send back: the two STOP decisions with the diagnostic and Figure 1 that
justified them, and the per-experiment RESULTS.md files. The paper's
results section is written from those.

---

## 9. Later: replicating the campaign with live teacher models

Tier 4 already ran the five E1-style controllers with four live teachers
(§1.3). A full live replication is possible but is a second, separate
table — never pooled with the cache-mode figures — because the general and
code live teachers are stand-ins for the models that wrote the cached text.

- **Cost:** ~30 GB and ~2× wall clock per run (four teacher forwards per
  step); `CONCURRENT=2–3`. Full campaign ≈ 300 GPU-hours; E1 + E5 only
  ≈ 55 runs, the sensible scope.
- **How:** copy the manifest, add the `live: &live` anchor from
  `configs/experiments/live_kd_small.yaml` (`teachers.mode=live`,
  `teachers.num_teachers=4`, `distillation.lambda_ce=0.5`,
  `++data.router_strict=true`), splice `- *live` into every spec's
  overrides, rename `experiment:` with a `_live` suffix. Health check at
  step 0: `loss/kd` > 0 and `loss/kd_row_hit_frac` = 1.0.
- **If the teachers must score their own text everywhere:** regenerate
  `teacher_output_text` for general and code with the live models first.
  That is a new dataset and a new stream digest, so it is a new campaign,
  not a rerun.
- **Do this after** E5, E4 and the ablations are in and the paper's
  cache-mode figures are final.

## 10. Known gaps (do not fix during the campaign; report them)

- `stable_gate` (STABLE, probe-gated merge) is not implemented; reviewers
  may ask (STATUS §5).
- Math/medical teacher-text provenance rests on ids and logit statistics;
  the generation logs are still to be obtained (`docs/data_provenance.md`).
- Paper wording fixes are listed in STATUS §7 (sequence-level
  distillation, confidence semantics, tracked moments, truncation, no
  teacher-selection claim).
- Consolidation happens mostly through the 250-step sweep re-validating F
  modules, not only through P decisions (138 merges vs 103 P decisions in
  the live run); the analysis reports both, and the paper must say so.

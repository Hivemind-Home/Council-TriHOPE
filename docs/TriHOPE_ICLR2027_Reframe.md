# TriHOPE — Reframe for ICLR 2027

**Status:** decision document for the team
**Deadline:** abstract Sep 18, paper Sep 25, 2026 (AoE). 23 days from today (Sep 2).

---

## 0. The vision, in plain words (read this even if you read nothing else)

Here is what we are doing and why, without the paper language.

We are not going to win by training a small model and beating other small models on a benchmark. Everyone with more GPUs and more data does that better than us, and our own method holds updates back on purpose, so on a raw score we will always look worse than someone who just trains on everything. If we submit "our student is better," we lose. So we stop chasing that.

What we can show is that something fundamental works. Today, when a model learns from a stream, every gradient goes straight into the weights. Nobody asks whether that update deserves to be permanent. A one-off example, a noisy teacher, a pattern that has not settled yet — all of it gets written into the same place with the same weight as a rule the model has seen a hundred times and learned consistently. Our idea is that the optimizer already knows the difference. Adam's running averages tell you whether this gradient is unusual, whether it agrees with what came before, and whether the same pattern keeps coming back. We read that, for free, and use it to decide where each update should live: nowhere yet (park it), in a reversible scratch layer (LoRA), or in the base model for good. And we make that decision real at the optimizer level, so a "closed" weight does not move at all, not even by weight decay.

The second half of the vision is what this buys you beyond less forgetting. Because every write is scoped to a block, tagged with the teacher and the evidence that caused it, and every permanent merge is checkpointed on its own, the model has a complete history of who changed what and why. That means if a teacher turns out to be bad — wrong answers, low confidence, or deliberately poisoned — most of its influence never reached the base weights in the first place, and whatever did can be found and reverted without throwing away everything learned afterward. No other method can do that, because once a gradient goes through normal Adam it is entangled with everything else forever.

So the paper's job is to prove those two things on real data: that reading the optimizer beats simpler triggers on the forgetting-versus-learning trade-off, and that every permanent change to the model is attributable and reversible. That is a mechanism paper, not a leaderboard paper. Reviewers accept mechanism papers when the control experiments are clean. That is the whole plan below.

---

## 1. The decision in one paragraph

We stop presenting TriHOPE as a way to make a small student model better through multi-teacher distillation. We cannot win that game: any controller that withholds updates will lose to plain full-parameter distillation on raw student quality, and our own Table 1 already shows it (36% vs 100% on the new domain). Distillation is only the source of the gradient stream. The paper is about **what happens to an update after it is computed**: whether it is discarded, held reversibly, or made permanent. We claim that this decision can be read off Adam's pre-update state for free, enforced exactly at the optimizer, and that it produces a better forgetting–plasticity trade-off than the triggers people currently use. Every sentence in the paper that implies "the student gets better" goes. Every result is a curve or a controlled comparison between *controllers*, not a leaderboard number.

---

## 2. What we are and are not claiming

**We are claiming:**

1. Permanence of an update is a decision the optimizer does not currently make, and it can be made per block from pre-update `m`, `v`, and the current gradient at no extra cost.
2. A three-tier write policy (R = defer/write nothing, F = tentative LoRA, P = permanent base) traces a better forgetting-vs-plasticity curve than two-tier and heuristic-trigger alternatives at a matched permanent-write budget.
3. The no-write tier is useful only because deferred observations can come back (replay into F on recurrence). Deferral is quarantine, not deletion.
4. F→P promotion gated on stability ∧ recurrence fires at sensible times, and mistimed promotion measurably hurts.
5. MaskedAdamW makes the routing boundary exact: closed coordinates see no parameter, moment, weight-decay, or bias-correction change (Theorem 1). This is a correctness property no other routing method has.
6. **Every permanent change is attributable and revertible.** Because writes are block-scoped, teacher-tagged in the event trace, and P-merges are individually checkpointed, θ_T − θ_0 decomposes exactly into logged, teacher-attributed writes (a corollary of Theorem 1: ledger completeness). A bad teacher's influence is mostly quarantined in R/F, and what reached P can be reverted selectively. No baseline can do this.

**We are not claiming:**

- That TriHOPE produces a stronger 0.6B/1.7B model than distillation baselines on standard benchmarks.
- That teacher routing (cosine prototypes) is a contribution. It is behind PerSyn/H-OPD and stays in the setup section.
- That the LoRA→base merge is a theorem worth headlining. Theorem 2 is the LoRA identity; keep it in the appendix as a sanity property.
- "Optimizer as memory." Nested Learning/HOPE owns that phrase. We say "optimizer state as evidence for permanence."

**Working title:** *Which Teacher Wrote This? Evidence-Gated, Attributable Routing of Distillation Updates Across Deferred, Fast, and Permanent Memory*

**Why multi-teacher stays in the paper:** it is the reason the problem exists. Several teachers with uneven quality is the realistic setting where an update can be correct for one sample yet wrong to make permanent, and where "who wrote this into the weights" is a question anyone deploying the model will ask. Teacher *selection* is still not our contribution; teacher *accountability* is.

---

## 3. Why the current draft fails review

Four problems, all fixable. They are listed in order of severity.

**3.1 The R tier is write-only, so our forgetting number is a confound.**
`writer._execute_r` appends an embedding and returns. Nothing reads the store back except a hit-rate metric. So TriHOPE forgets less on domain A because it never learns domain B. A reviewer will say "less forgetting because less learning" and be right. The discussion section admits this; admitting a confound does not remove it.

**3.2 The only evidence is a 48-token, width-32 toy.**
There are no `runs/` on the real stream. The infrastructure (stream configs, manifests, preflight, analysis tables, 400+ tests) is ready but has not been executed. ICLR's CFP this year explicitly says to keep working rather than submit something incomplete.

**3.3 P never fires.**
Default thresholds (`min_stability_C=0.7`, `min_repetition=0.6`, `stability_mode=instant`) make consolidation a null result. The draft then evaluates consolidation by a 1.2e-7 logit-invariance check, which is not a result.

**3.4 MoLF (arXiv 2605.07111, May 2026) is not cited and is the closest prior work.**
It routes updates between an FFT expert and a LoRA expert per module using an Adam-moment score, applies AdamW only to Top-K winners, and leaves losers untouched. That is our F-vs-P decision plus our masked optimizer. Our differences are real (a no-write tier, task-free streaming with forgetting as the target, consolidation on stability ∧ recurrence, coordinate-local bias correction) but they must be stated in the first paragraph of related work, not discovered by the reviewer. Also cite arXiv 2604.22407 (OGP, Adam second-moment pathway in continual learning).

---

## 4. Positioning against prior work

| Prior work | What it decides | What it lacks vs. us |
|---|---|---|
| MoLF (2026) | LoRA vs FFT per module from Adam score, Top-K sparse AdamW | No no-write tier; no consolidation; no forgetting objective; shared bias-correction clock |
| Online-LoRA / STABLE | When to merge LoRA into base | Trigger is loss plateau or a probe, not optimizer state; two-tier |
| Titans / Nested Learning (HOPE) | Surprise-gated memory writes; multi-timescale fast/slow | Surprise alone, no stability or recurrence; memory is a learned module, not a write policy over existing weights |
| OGP (2604.22407) | Projects gradients while respecting Adam's moment pathway | Modifies the gradient; does not decide permanence |
| Gradient Routing | Which params a gradient may touch | Static mask, not evidence-driven, no promotion path |
| Sparse memory finetuning (Lin et al. 2025) | Which memory slots to update | Single tier |
| HippoRAG 2 / RAG | Keep information external | No rule for when external becomes parametric |

The one-line gap: every rival is two-store and decides *where* once. We are three-store, decide *when to promote*, enforce it exactly, and can say afterwards which teacher wrote what and undo it.

---

## 5. Experiments that must exist

All on `stream_small` (Qwen3-0.6B, 6000 steps, 7 phases), 3 seeds (`1337, 2024, 7`). Headline 1.7B run only if time remains.

### E1 — Controller comparison (make-or-break)

Same stream, same student, same LoRA budget, only the write policy varies.

| Spec | What it is | Status |
|---|---|---|
| `trihope` | full R/F/P from m/v | exists |
| `full_ft` | everything trains | exists |
| `lora_only` | all adapters always open | exists |
| `no_retrieval`, `no_consolidation` | ablations | exist |
| `molf_style` | two-tier F/P from an Adam score, no R, no merge | **add** |
| `plateau_trigger` | Online-LoRA-style: write F always, merge to P on loss plateau | **add** |
| `surprise_gate` | Titans-style: route on S alone (R if high, F otherwise) | **add** |
| `random_routing` | same per-phase action shares as `trihope`, assignment shuffled | **add** |

`random_routing` is the control that proves the signals carry information. If it matches `trihope`, the signals do nothing.

**Pass condition:** `trihope` dominates `molf_style`, `plateau_trigger`, and `surprise_gate` on the forgetting-vs-new-domain curve at matched permanent writes, and beats `random_routing` clearly. If it does not, see §9.

### E2 — The no-write tier

`R+F+P` vs `F+P` vs `P-only`, with R **replay enabled** (see §6.1). Report new-domain accuracy, old-domain retention, and steps-to-recover on `code_revisit`. The claim is that deferral improves the curve, not that retrieval helps generalization.

### E3 — The budget curve (the paper's identity figure)

x-axis: number of permanent writes (or active fraction). y-axis: worst-case retention delta. One curve per controller, swept by threshold. This is Figure 1. Everything else supports it.

### E4 — Consolidation timing

Use `p_study_small.yaml` with `stability_mode=sustained` and the lowered thresholds so P fires. Report when P fires and the signal values at those moments (`p_selection_stats`), plus the three forced-merge damage runs (`p_forced_bad`, `p_forced_plausible`, `p_forced_lowconf`) with recovery half-life. The result is "well-timed merges are free, mistimed ones cost X and take Y steps to recover."

### E5 — Bad-teacher attribution and rollback (the uniqueness pillar)

Corrupt one teacher for one recurrent phase: shuffle or degrade `teacher_output_text` on a fraction (25–50%) of math rows and set their `teacher_confidence` low so the gate is exercised. Run `trihope`, `full_ft`, `lora_only`, and `molf_style` on that stream. Report:

1. **Containment** — from the ledger, what fraction of the corrupted teacher's actions ended in R, F, and P.
2. **Damage** — retention delta on general/code/medical caused by the corrupted phase, per method.
3. **Selective rollback** — revert only the P-merges attributed to the corrupted teacher via their pre-merge checkpoints (`checkpoint_before_merge=true`), keep everything learned afterwards, and measure recovery. Full FT's only option is a full checkpoint restore, which discards later learning; report that cost.

**Pass condition:** most corrupted influence stays out of P, TriHOPE's damage is lower than full FT and MoLF-style at matched budget, and selective rollback recovers more than a full restore. Even if E1 is a tie on the Pareto curve, this result stands on its own.

### Metrics (report all, everywhere)

- Per-domain retention delta at every phase boundary (worst-case and mean)
- New-domain loss/accuracy at end of each recurrent phase
- Anytime/online loss (area under training loss per phase)
- Permanent-write count, active coordinate fraction
- Steps-to-recover on `code_revisit`
- Wall-clock and controller-overhead fraction (`run_summary.json`)

---

## 6. Code changes required

### 6.1 R → F replay promotion (blocks E2 and fixes §3.1)

When `RetrievalHitRepetition` records a hit on a bucket that already has entries in the R-store, replay the stored item (we already keep `teacher_output_text` and `bucket_id` in `RetrievalEntry`) as an F write on the next step. Minimal version: on hit count ≥ `replay_threshold`, add the stored sample to the consolidator's replay buffer and open the block's Top-K adapter. Add a config flag `controller.retrieval.replay_on_hit` so the old behavior stays as an ablation (`r_terminal`).

### 6.2 Baseline specs (blocks E1)

Add to `configs/experiments/baselines_small.yaml`:
- `molf_style`: `controller.ablation.disable_stores=[R]`, consolidation disabled, F/P decided by `stability_adam` (the m²/v ratio we already compute) above a threshold. Needs a small `policy.mode=adam_score` branch.
- `plateau_trigger`: `controller.enabled=false`, `train.trainable=lora`, plus a merge hook that fires on N-step loss plateau. Reuse `ConsolidationScheduler` with a plateau condition instead of signal re-validation.
- `surprise_gate`: `controller.ablation.disable_signals=[repetition, cosine, volatility]` — the existing neutral-pinning already reduces the policy to the surprise test.
- `random_routing`: `debug.policy_override=random_matched` — new override that samples store labels from the running per-phase share of `trihope`.

### 6.3 Make P fire

Set `controller.consolidation.stability_mode=sustained` as the default for the stream configs, with `period=250`, `min_stability_C=0.4`, `min_repetition=0.4` (the `p_study` values). Keep the strict values as an ablation.

### 6.4 Bad-teacher stream and selective rollback (blocks E5)

- A `data.corrupt_teacher` config block: `{domain, phase, fraction, mode: shuffle|degrade, confidence: 0.2}` applied at load time so the corrupted rows are deterministic per seed and recorded in the schedule digest.
- `checkpoint_before_merge=true` in the E5 specs; tag each pre-merge checkpoint with the teacher name already present in the consolidation event.
- A small `analysis.tables.teacher_attribution` that reads `events.jsonl` + `ledger.pt` and reports per-teacher R/F/P counts and coords written to P.
- A `hivemind rollback --teacher <name>` command (or script) that restores the pre-merge checkpoint for each P-merge attributed to that teacher and replays the subsequent event log with those merges skipped. Minimal version for the deadline: restore the earliest attributed pre-merge checkpoint and re-run the stream from there with `disable_stores=[P]` for that teacher.

### 6.5 Answer the cos(g, m) noise critique

Reviewers familiar with the Adam-pathway literature will say the instantaneous cosine is noisy. We already track the sustained EMA `C̄`; use it in the policy, report both, and add a one-line ablation.

---

## 7. Writing changes

**Abstract, first sentence:** "Optimizers decide how large an update is; nothing decides how permanent it should be."

**Introduction:** open with the stream problem (a good teacher's gradient can still be too rare, too unstable, or too local to commit). State the three tiers as increasing commitment. State the contribution list from §2 verbatim. One sentence on distillation as the setting.

**Related work, paragraph 1:** MoLF, then Online-LoRA/STABLE, then Titans/HOPE, each with the one-line gap from §4. Do not bury this.

**Method:** keep §4.1–4.4 of the current draft (they are correct). Move Theorem 2 and Proposition 1 to the appendix. Keep Theorem 1 and Corollary 1 in the main text as one half-page block, and add **Corollary 3 (ledger completeness):** under MaskedAdamW, θ_T − θ_0 equals the sum of authorized, logged writes, each tagged with the teacher and signal values that caused it. One paragraph, follows directly from Theorem 1.

**Results:** Figure 1 is the budget curve. Table 1 is E1 at a fixed budget. Table 2 is E2. Figure 3 is action composition over the stream (exists). §E4 is one table plus one recovery plot. **Figure 4 is E5:** containment bars per method, damage, and the selective-rollback vs full-restore recovery curves. The security paragraph moves from Limitations to Contributions.

**Cut list:** "improves distillation outcomes," "optimizer-as-memory," "function-preserving" as a contribution, the 48-token toy (move to appendix as a mechanism sanity check, or drop), the runtime paragraph (replace with the overhead fraction from `run_summary.json`).

**Limitations to state plainly:** thresholds are hand-set; student is ≤1.7B; teachers are cached traces (one live-KD run at most); retrieval is not consulted at inference; signals inherit Adam's coordinate scaling.

---

## 8. Timeline

| Dates | Work | Owner |
|---|---|---|
| Sep 2–5 | §6.1–6.4 code changes, tests, `stream_smoke` pass, preflight on all new specs | eng |
| Sep 5–12 | E1 on `stream_small`, 3 seeds, 9 specs (~27 runs, parallelize across GPUs) | eng |
| Sep 9–13 | E4 (`p_study_small`) in parallel | eng |
| Sep 6–8 | §6.4 corrupt-teacher loader + attribution table + rollback script | eng |
| Sep 10–15 | E5 bad-teacher runs (4 specs × 3 seeds) + rollback | eng |
| **Sep 13** | **Go/no-go on E1 (see §9)** | all |
| Sep 13–16 | E2, E3 threshold sweep, `analysis.run_report`, figures | eng + writing |
| Sep 14–18 | Abstract, intro, related work, method rewrite | writing |
| **Sep 18** | **Abstract deadline** | |
| Sep 18–23 | Results, discussion, limitations, appendix; 1.7B headline only if GPUs are idle | all |
| Sep 24 | Internal read-through against §2 and §7 cut list | all |
| **Sep 25** | **Paper deadline** | |

---

## 9. Go/no-go criteria

On Sep 13, after E1:

- **Go:** `trihope` is on or above the Pareto front of `molf_style`, `plateau_trigger`, and `surprise_gate` at matched permanent writes, and clearly separates from `random_routing` (non-overlapping error bars on at least worst-case retention).
- **Conditional go:** E1 is a tie with the trigger baselines but E5 shows clear containment and working selective rollback. Submit with attribution/rollback as the headline and the trade-off as "matched at no cost."
- **No-go:** a baseline dominates us on E1 **and** E5 shows the corrupted teacher reaching P at a similar rate to the baselines. In that case we do not submit a toy-only paper to ICLR. We submit the mechanism paper to a workshop, use the 1.7B run and the R-replay results to fix whatever E1 exposed, and target ICML (abstract Jan 16, paper Jan 22, 2027).

Either outcome is fine. Submitting the current draft is not.

# TriHOPE — The Reframe (one page)

**Thesis:** stop selling "optimizer-as-memory." Sell the missing axis of
optimization: **permanence as a budgeted, optimizer-driven decision.**

## Where we are now
- **Setting:** task-free **online continual learning** — *not* pretraining,
  *not* fine-tuning. The multi-teacher distillation is only the supervision
  source, not the problem.
- **What we've been claiming:** optimizer-state-as-memory; surprise /
  stability / repetition signals; tri-store R/F/P routing.
- **Why that fails:** every piece is already owned — Nested Learning/HOPE
  (optimizer-as-memory; multi-timescale fast/slow), Titans (surprise), GSNR
  (m²/v), Cockpit/TRACE (training-time interpretability), Gradient Routing
  (route the update), Online-LoRA + STABLE (signal-timed LoRA→base merge),
  MoLF (route by Adam m/v). As framed today, a reviewer files us under
  **"HOPE + a router."**

## The reframe
Optimizers control **magnitude** (the learning rate). Nobody controls
**permanence**. We make permanence a **controlled resource, allocated by
evidence:**
- **R** = *defer* — write nothing to the weights (quarantine a one-off).
- **F** = *tentative* — reversible LoRA fast weights.
- **P** = *permanent* — consolidate into base weights.
- The **P-write rate is a budget**, the optimizer's **m/v is the allocator**,
  and the narrative is **systems consolidation** — the brain does not
  consolidate everything; consolidation is costly and gated.

## What is actually ours (the one defensible claim)
A task-free controller that reads pre-update Adam-style **m *and* v per
block**, routes each update across a **three-tier store whose R tier writes
nothing** (every rival is two-store), promotes **F→P only under joint
stability ∧ recurrence** (the opposite trigger to Online-LoRA's novelty
signal), **without touching the update rule**, with **exact coordinate
masking**. The rigor that is genuinely ours is the exact masking — **not**
the merge.

*Correction (2026-09-05, first real run):* under exact masking the
optimizer's own m/v of a never-opened coordinate are zero forever, so
reading them literally makes surprise saturate and C̄ = 0 on every
base-weight module, and P can never fire. The controller therefore keeps
its own bias-corrected Adam-style moments for every indexed coordinate
(MoLF's universal momentum tracking, confined to the signal path; one EMA
pair on a 1/8 coordinate sketch, ≈ 0.6 GB for 0.6B). They equal Adam's
state on coordinates that are open every step and stay alive on closed
ones. "Zero optimizer overhead" becomes "no change to the update; a
signal-side EMA"; Theorem 1 is unchanged because it is about the update.

## What we do (in priority order)
1. **E1 — the make-or-break run.** m/v-timed routing vs **Online-LoRA**
   (loss-plateau trigger), **STABLE** (probe gate), and a **Titans-surprise**
   baseline, on the *same* stream. If reading m/v beats them, we have the
   paper. This is the whole bet.
2. **E2 — the no-write tier earns its place.** R+F+P vs F+P vs P-only. Frame
   R as **deferral / quarantine**, not generalization (retrieval helps
   recurring items, not the novel one-offs we send to R).
3. **E3 — the budget curve.** forgetting vs. number-of-permanent-writes;
   show m/v allocates the budget better than heuristic triggers. This figure
   is the paper's identity.

## Must-drop / must-fix
- **Drop** "function-preserving merge" (it is the exact LoRA identity) and
  "optimizer-as-memory" (HOPE owns it).
- **Answer** "isn't surprise just Titans'?" in paragraph 1 of related work;
  **rebut** GMC's `cos(g,m)`-is-noisy critique (adopt `g·m/v`, or defend the
  sustained EMA-of-C as the fix).
- **Add** a plasticity probe (dormant units / effective rank) and
  anytime/online accuracy; **check MoLF's date** against our first public one.
- **Scope out** teacher-routing as a contribution (our cosine router is
  behind PerSyn/H-OPD).

## Verdict
As-is: an honest recombination → workshop / mid-tier. Reframed, with E1–E3
and one mid-scale run: a **new axis of control** (permanence as a budgeted,
optimizer-driven decision) → top-tier-capable. **Everything rides on E1.**

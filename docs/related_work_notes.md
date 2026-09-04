# Related-work reading notes (Part C)

One section per paper, fixed structure (Claim / Mechanism / Overlap with
TriHOPE / Our difference / Baseline implication / one sentence / BibTeX).
Every fact was read from the arXiv abstract or HTML/PDF on 2026-09-04;
section numbers refer to the HTML versions. BibTeX entries are collected
in `paper/refs.bib`. Papers that could not be fetched are marked as such
rather than summarised from memory.

TriHOPE, for reference: an online continual-distillation controller that,
after backward and before `optimizer.step`, reads each block's gradient g
and pre-update Adam moments (m, v) — surprise S = mean(g²/(v+ε)),
directional stability C = cos(g, m) and its EMA C̄, ‖g‖ volatility,
recurrence R — and routes the block's update to R (write nothing; park the
row, replay it into LoRA when its bucket recurs), F (Top-K LoRA rank
components) or P (base weights; LoRA→base merge on sustained stability ∧
recurrence). `MaskedAdamW` leaves closed coordinates exactly untouched.
Every write is teacher-tagged and every merge is checkpointed, so permanent
writes are attributable and revertible.

---

## Paper 1 — MoLF

Two facts up front: **MoLF's EPD is not m²/v** — our earlier `stability_adam`
is (up to ε placement and squaring) MoLF's *ablation baseline* PFN (App. A.3),
so the `molf_style` spec now uses MoLF's own EPD argmax rule
(`policy.adam_score_rule=epd_argmax`). Everything below is from the PDF.

### MoLF — Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation (arXiv:2605.07111, 2026)

First arXiv submission: 8 May 2026 (v1). Authors: Haozhan Tang, Xiuqi Zhu,
Xinyin Zhang, Boxun Li, Virginia Smith, Kevin Kuo.

- **Claim.** Neither FFT nor LoRA is structurally limited; the right choice
  is per-module and can be made online by the optimizer from its own Adam
  moments. Routing each linear module's update to the "expert" (dense W or
  LoRA pair) with the largest expected loss reduction per parameter
  recovers the better of FFT/LoRA within 1.5 % in all 9 (model, task) cells
  and wins outright in 3 (Sec. 5.2, Table 2).
- **Mechanism.** Architecture (Sec. 4.2, Eq. 1): every linear layer is an
  ungated superposition `y = W_base x + Σ_i (α_i/√r_i) B_i A_i Dropout(x)`;
  expert 0 is `W_base`, each `(A_i, B_i)` a LoRA expert; all experts see
  every token every step, sparsity is "strictly deferred to the optimizer".
  Phase 1, universal momentum tracking (Sec. 4.3, Eqs. 2–3): standard Adam
  EMAs for **every** expert, winner or loser, on **one shared step
  counter** ("dormant experts maintain a mature, debiased momentum state").
  Phase 2, EPD score (Eq. 4; App. A.1): `S_t^(i) = (η_t^(i) / N_params^(i))
  · Σ_θ (m_t^(i))² / (√v_t^(i) + ε)` with uncorrected post-update moments,
  a per-expert learning rate, ε = 1e-8; units of loss decrease per
  parameter, deliberately not scale-invariant. Phase 3 (Eq. 5): per module
  the Top-K (K = 1 everywhere) experts get the full AdamW step (λ_FFT = 0.1,
  λ_LoRA = 0.01); losers keep `θ_t = θ_{t-1}` but their moments and counter
  keep advancing. Routing every step, independently per projection matrix.
  Fusion (Sec. 4.5): once, post-training, `W_final = W_base + Σ_i (α_i/√r_i)
  B_i A_i`; no online merge. Findings: dense (no routing) updates collapse
  Med on Qwen-3B 60.60 → 31.08 (Table 3); EPD ≥ PFN on 5/6 cells, largest
  +2.58 on Med (Sec. 5.4.2); modules "definitively specialize in either FFT
  or LoRA early in training" (Fig. 4). Setup: Gemma-3-1B, Qwen2.5-1.5B/3B;
  CounterFact, MedMCQA, text-to-SQL; single-task SFT. Absent: continual
  learning, forgetting metric, retrieval/replay, per-coordinate masks inside
  an expert, provenance/revert.
- **Overlap with TriHOPE.** Reads Adam m, v per module after backward and
  before the step to decide where the update lands; two persistence tiers
  per module (base vs LoRA = our P vs F); a masked AdamW that updates only
  the chosen tier; a per-module aggregate of `m²/(√v+ε)` on the same
  tensors as our `stability_adam`; LoRA merged into base at the end; per-
  module routing every step; their Table 3 already shows "routing beats
  dense updating" and Fig. 4 the early persistent specialization.
- **Our difference.** (1) A no-write tier: MoLF always writes somewhere;
  our R parks the row and replays only on recurrence (E1 at matched
  permanent writes; E5 where MoLF has no place not to commit). (2)
  Permanence as an online, gated, revertible decision: MoLF merges once,
  unconditionally, post-training; ours per block on sustained stability ∧
  recurrence, each merge checkpointed and teacher-tagged (rollback
  experiment). (3) Exact masking: MoLF's losers accrue m, v on the shared
  clock; ours have no parameter, moment, weight-decay or bias-correction
  change (coordinate-local counters). (4) Sub-expert granularity in F
  (Top-K rank components). (5) Signals beyond the Adam ratio and a
  stream/forgetting evaluation.
- **Baseline implication.** `molf_style` now implements EPD faithfully:
  per block the dense expert (base) and the LoRA expert compete on
  `lr_i · mean(m²/(√v+ε))` (`ModuleSignals.epd_score`, pre-update moments —
  one EMA step behind MoLF's post-update ones), the winner alone updates
  (P = base fully, F = the whole adapter with `top_k_fraction=1.0`), every
  block routes every step, no R, no merge. **Not reproduced:** universal
  momentum tracking (losers' moments advancing on a shared clock) — that
  contradicts Theorem 1 and MaskedAdamW; documented as the one deliberate
  deviation. The PFN-like SNR threshold survives as `adam_score_rule=
  snr_threshold` (sweep specs `molf_style_a*`). MoLF's own numbers do not
  transfer (single-task SFT).
- **One sentence.** MoLF (Tang et al., 2026) reads each module's Adam
  moments to route every update to either the dense weight or a LoRA
  expert via an expected-preconditioned-descent score under a Top-1
  masked AdamW, and fuses the adapters once after training; it decides
  *where* an update lands but never *whether* to write, updates all
  experts' moments on a shared clock, and has no online, gated or
  reversible path to permanence.
- **BibTeX:** `tang2026molf` in `paper/refs.bib`.

**MoLF score vs TriHOPE stability_adam: different.** EPD is `(η/N) Σ
m_t²/(√v_t+ε)` (post-update, per expert, argmax); `stability_adam` is the
dimensionless `mean(m_{t-1}²/(v_{t-1}+ε))` ∈ [0,1] — the square of MoLF's
PFN baseline up to ε. The `molf_style` spec was relabelled and re-specified
accordingly (T9).

## Paper 2 — OGP

### OGP / Adaptive-OGP — Hidden Failure Modes of Gradient Modification under Adam in Continual Learning, and Adaptive Decoupled Moment Routing as a Repair (arXiv:2604.22407, 2026)

v1 24 Apr 2026; **v2 (24 Jul 2026) is a withdrawal** ("substantial issues
in the experimental design and evaluation that affect the validity of the
main conclusions"). Authors: Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu,
Li Song. Everything below is from v1; its numbers must not be cited.

- **Claim.** Continual-learning methods that shrink the gradient in
  protected directions and hand the shrunken gradient to *both* Adam
  moments ("shared routing") silently undo their own protection, because
  Adam's denominator shrinks in the same directions; feeding the modified
  gradient only to m while v sees the raw gradient, with an overlap-
  adaptive strength, avoids the collapse.
- **Mechanism.** Projection family `g_mod = g − α U Uᵀ g` (covers OGD, GPM,
  SGP, ROGO, TRGP, FOPNG, Adam-NSCL), penalty and replay-mixing families
  (Sec. 2). Proposition 1 (Sec. 3, Eq. 1): along a protected direction u,
  steady-state `v_∞(c) = c² E[(uᵀg)²]`; shared routing gives c = 1−α, so the
  effective step inflates by exactly `1/(1−α)` (β2 drops out), and because
  m_u shrinks by the same factor the net Adam step along u is essentially
  unchanged — the protection is inverted. Repair (Sec. 4, Alg. 1):
  `s_t = ‖Uᵀg_t‖²/‖g_t‖²`, EMA `s̄_t` (β_s = 0.99), `α_t = α_max(1−s̄_t)`,
  modified gradient → m only, raw gradient → v, standard bias correction;
  U from a randomized SVD of a 50-step gradient buffer per parameter group,
  refreshed every 10 steps. Backbone: a 256M HOPE. Absent: tiers, stores,
  merges, per-coordinate masking or counters.
- **Overlap with TriHOPE.** Same top-level thesis (the moments are where CL
  decisions are made or undone); same backbone family and forgetting
  metric shape; a gradient/subspace-alignment signal with an EMA of which
  our `cos(g, m)` + EMA is the rank-1 special case; routing between what
  enters m and what enters v (our MaskedAdamW is the α = 1 endpoint of
  shared routing); and their diagnostic applies to our own surprise
  `S = mean(g²/(v+ε))` — any TriHOPE mechanism that *attenuates* rather
  than masks would deplete v and inflate both the step and S.
- **Our difference.** Masking, not attenuation: closed coordinates receive
  no moment update at all, v is frozen rather than decayed to (1−α)²v, and
  bias correction restarts from a coordinate-local counter — so the
  1/(1−α) inflation cannot occur on reopening (`tests/test_masked_adamw.py`;
  a cheap "scale g by (1−α) instead of masking" ablation would reproduce
  their fingerprint in our code). We route updates across stores; they
  route one gradient between two moments and every parameter still moves
  every step. No SVD buffer. We decide permanence with attribution and
  revert; nothing in OGP is revertible.
- **Baseline implication.** Not a store router; `molf_style` does not cover
  it. Cite v1 as a withdrawn preprint for Proposition 1 only, as motivation
  for exact masking; do not report its numbers. A shared-routing projection
  baseline (GPM/OGD-style on `full_ft`) is only worth adding if a reviewer
  asks.
- **One sentence.** Hu et al. (2026; withdrawn preprint) show that any
  gradient attenuation fed to both Adam moments inflates the effective
  step along the attenuated directions by 1/(1−α) through the second-
  moment denominator; this is the analysis that motivates TriHOPE's exact
  coordinate masking — closed coordinates receive no moment update rather
  than a scaled one — while their method still updates every parameter
  every step and has no notion of a write that is withheld, parked, or
  made permanent.
- **BibTeX:** `hu2026attenuate` (with the withdrawal note).

## Paper 3 — Gradient Routing

### Gradient Routing — Masking Gradients to Localize Computation in Neural Networks (arXiv:2410.04332, 2024)

- **Claim.** Data-dependent masks on the backward pass let a user decide
  which data updates which subregion of a network without changing the
  forward pass or the loss; localised regions can be ablated for robust
  unlearning, and routing a narrow labelled subset "absorbs" the broader
  capability (Abstract; Sec. 3, 5).
- **Mechanism.** For each data point a user-supplied route `{α_e}` over
  edges of the computation graph multiplies the chain-rule terms; masks
  sit on activations of a few layers, mostly binary
  (`x = mask*act + (1-mask)*act.detach()`, Alg. 1); the forward pass is
  unchanged and "any gradient-based optimizer, like SGD or Adam" is used —
  the paper never claims closed parameters are left untouched by the
  optimizer. Routes come from membership in a coarse partition (forget vs
  retain); for TinyStories a per-token convex combination weighted by
  forget-vs-retain frequency (App. C). ERA (Sec. 4.2.2): **Expand** (add 64
  MLP neurons in layers 0–4 of a 28M Transformer), **Route** (train from
  scratch; on routed forget tokens the mask on the original dims is
  **−0.75**, plus an L1 penalty on target-layer activations), **Ablate**
  (delete the added neurons, fine-tune on 64 retain stories). Absorption
  (Sec. 5): routing a subset localises the broader capability (California
  → Oregon/Colorado/Texas; WMDP-bio +0.182 loss after ablation vs
  FineWeb-Edu +0.032, Table 1); did not hold for DEMix. Baselines: data
  filtering (re-initialise and retrain on retain data — the "gold
  standard"), RMU, DEMix + ablate; with < 100 % of forget stories labelled
  ERA dominates even filtering on unlearning and robust unlearning at a
  retain-loss cost proportional to the routed data (Fig. 4).
- **Overlap with TriHOPE.** Every ingredient of our `gradient_routing`
  baseline and some of TriHOPE proper: a per-sample mask applied between
  backward and the step; confining a source to a subregion that can be
  excised (our revert / slice zeroing is their Ablate); masked training
  with Adam on added dimensions (LoRA rank = "added dimensions"); a post-
  ablation retain fine-tune (our damage/recovery); the vocabulary of
  provenance-based localisation (their App. J classifies DEMix as
  "provenance labels → expert"); a partial-labelling sweep (the bad source
  only partly known). Their unlearning results are stronger than a rank-
  slice ablation will be, because ERA trains from scratch with a negative
  mask.
- **Our difference.** Their mask is a user-supplied function of the label;
  our decision reads only the block's gradient and moments and never sees
  the teacher label, which is attached *after* the write for provenance.
  Their masks are on activations and interact with Adam only implicitly;
  MaskedAdamW leaves closed coordinates and their m/v untouched
  (`tests/test_ledger_completeness.py`). They have no permanence tier;
  our P tier makes permanence a gated, checkpointed, revertible event. They
  require Expand before training from scratch; we run on a pretrained
  student in an online stream. Absorption is a property they want and a
  threat for us (a corrupted teacher's features absorbed into a clean
  slice) that provenance tagging is designed to detect.
- **Baseline implication.** Yes — `gradient_routing` in
  `bad_teacher_small.yaml`: Expand ≈ the LoRA rank (16 split into 3-
  component slices per teacher, 5 teachers, one component unused), Route ≈
  open only the batch teacher's slice, base closed, no merge (a binary
  parameter mask), Ablate ≈ zero the corrupted teacher's slice at step
  3649. State the differences: ERA's α = −0.75 on the original dims plus
  an L1 penalty vs our {0,1} and no penalty; activation vs parameter masks
  (our closed slices are exactly frozen); from-scratch pretraining vs online
  fine-tuning; per-token vs per-batch labels; recovery = the continuing
  stream, not a 64-sample retain fine-tune. **Why data filtering is
  unavailable to us:** their gold standard re-initialises and retrains on
  the filtered corpus, which presupposes the forget label at ingestion; in
  an online stream the corrupted teacher is identified only after its rows
  were written, so the only filtering-shaped baseline is "restore the pre-
  corruption checkpoint and replay the stream without that phase" — our
  `full_restore` — which costs the stream again and is what provenance-
  tagged revert is meant to avoid.
- **One sentence.** Gradient Routing (Cloud et al., 2024) localises a
  labelled data source to a chosen subregion by masking the backward pass
  with a user-supplied, label-dependent route and then ablating the region
  (ERA); we adopt its per-source rank-slice partition as a baseline, but
  our routing decision never reads the label — provenance is recorded
  after the write, so a bad source can be excised without having been
  anticipated.
- **BibTeX:** `cloud2024gradientrouting`.

## Paper 4 — ReGrad

### ReGrad — Retrievable Gradients: Continual Post-Training Without Cumulative Weight Drift (arXiv:2606.15734, 2026)

- **Claim.** Treat a document's gradient as a retrievable "knowledge atom":
  precompute it offline into a Gradient Bank, retrieve query-relevant atoms
  at inference, apply them as a one-step *temporary* LoRA update, then
  revert — parametric knowledge injection with zero cumulative weight
  drift (Abstract; Sec. 1, 3).
- **Mechanism.** LoRA adapters in selected blocks, base frozen, gradients
  w.r.t. LoRA params only (Sec. 3.2). Stage 1, bi-level meta-learning: inner
  label-free LM loss on ≤ 200-token chunks, `θ_C = θ − α ⊙ g_C` with one
  scalar α per LoRA tensor; outer QA loss with the passages excluded from
  the prompt, so θ becomes a "gradient shaper" (Eqs. 2–5); without it raw
  gradients underperform direct generation (Sec. 5.3). Stage 2: per document
  store `g_i = ∇_θ L̃_{d_i}(θ*)` at the fixed meta-learned initialisation
  ("a few hundred KB" at 1B), indexed by text (BM25). Stage 3: retrieve
  top-K (K = 3), sum the atoms, `θ_q = θ* − α* ⊙ g(q)` (Eq. 6), generate,
  revert to θ*. Drift test (Table 2, LLaMA-8B): CPT average F1 30.73 →
  27.80 as 2.5K → 10K documents are injected; ReGrad 37.71 → 44.05. No
  deletion-as-unlearning, no online arrival beyond growing the bank.
- **Overlap with TriHOPE.** Adaptation confined to a LoRA subspace so
  per-item updates are cheap and reversible (our F tier); the thesis that
  permanent weight modification should be decoupled from knowledge
  acquisition (our reframe); a per-item store indexed for later lookup and
  replayed into LoRA (structurally our R store); replay as a literal
  optimizer-style update; a cumulative-drift analysis; the storage-cost
  argument we will need for R.
- **Our difference.** ReGrad is inference-side and ephemeral — nothing ever
  becomes permanent; our R entries are replayed at training time into F
  and can graduate to P. ReGrad stores gradients computed once at a frozen
  θ*; we store the row and recompute the gradient at replay time against
  the current weights. Their bank takes everything; our R decision is a
  per-block routing outcome. No provenance or revert (they never write).
  They need a supervised meta-training set; our replay uses the ordinary
  distillation loss. We can show a drift curve *with* a permanence tier —
  drift bounded by a budget rather than zero by construction.
- **Baseline implication.** Not a baseline (offline corpus, QA, no teacher,
  no online stream), but it forces a terminology fix. **Disambiguating
  sentence:** "Our R tier is a *deferral* store, not retrieval in the
  ReGrad sense: ReGrad retrieves precomputed gradients at inference time
  and applies them as a temporary, reverted weight delta, whereas our R
  tier parks a training row whose update was withheld and replays it into
  the LoRA adapter — permanently, at training time — only when its bucket
  recurs." Stop calling R "retrieval" in prose; keep the tier letter.
- **One sentence.** ReGrad (Su et al., 2026) removes cumulative drift by
  never writing at all — document gradients are precomputed into a bank
  and applied as reverted, query-time LoRA deltas — which is the
  inference-side complement of our training-time deferral tier; we instead
  make the permanent write a gated, attributable event rather than
  eliminating it.
- **BibTeX:** `su2026regrad`.

## Paper 5 — Attribution-Guided Continual Learning

### AGFT — Attribution-Guided Continual Learning for Large Language Models (arXiv:2605.05285, 2026)

- **Claim.** Layer-wise Relevance Propagation, extended from inputs to
  parameters, tells you which weight entries mattered for an old task;
  multiplying the new task's gradient by (1 − historical relevance)
  protects them and reduces forgetting better than EWC, replay or layer
  freezing (Abstract; Sec. 4.2).
- **Mechanism.** "Attribution" = **importance**, confirmed: LRP adapted to
  quantify the contribution of parameters to the predicted next-token
  logit (Sec. 4.1), element-wise relevance maps for FFN and attention
  weights (Props. 4.2–4.3) and for LoRA A/B; the words "provenance",
  "source", "data origin" do not occur. Per-task prior: train a task model
  first, compute per-sample relevance on correctly answered samples,
  log-scale, keep the mean of the top-K per entry (Eqs. 10–11). Gate:
  `R̄ = [max_{τ<t} R(W^(l,τ))]₊`, `∇̃L_t = (1 − R̄) ⊙ ∇L_t` (Eqs. 12–13),
  computed once per task offline; the optimizer is never named. Results:
  Concept-1K and TRACE on GPT-2 / Llama-3.2-3B, AGFT > EWC / SEQ / Replay /
  Freeze (Tables 1–2); EWC's failure attributed to Fisher being poorly
  estimated in LLMs.
- **Overlap with TriHOPE.** A per-parameter gate applied to the gradient
  between backward and the step (our insertion point); gates on LoRA A/B
  (our F coordinates); "which parameters may still move" decided from a
  scalar importance map; damping instead of freezing layers; and the
  finding that Fisher is a bad importance estimate in LLMs, which our
  "Adam v as a running Fisher proxy" framing must confront. Caveat: a
  constant per-coordinate gradient rescaling is largely cancelled by an
  Adam-family optimizer (m and √v scale together) except at exactly 0 —
  which is why TriHOPE masks the optimizer rather than the gradient.
- **Our difference.** Offline, task-boundary-dependent importance (a
  separately trained single-task model) vs online signals from the running
  optimizer with no boundaries; a gate that only protects vs routing in
  three directions including promotion and deferral; no record of which
  task wrote which entry (a max over tasks erases origin) vs a teacher tag
  per write and selective revert; soft gate vs exact zero-movement.
- **Baseline implication.** Not required (needs task boundaries and a
  per-task reference model); `adam_score` / EWC cover the importance-
  gating family if asked. **Disambiguating sentence:** "Attribution in Liu
  et al. (2026) is *importance* attribution — an LRP relevance score
  saying how much a weight entry contributed to a prediction — whereas
  attribution in TriHOPE is *provenance* attribution: a record of which
  teacher's write moved a weight, kept so that the write can be audited
  and reverted; the two are orthogonal, and one can compute the former over
  the latter's ledger."
- **One sentence.** Attribution-guided fine-tuning (Liu et al., 2026)
  gates each parameter's gradient by an LRP importance score computed
  offline per task; this is importance attribution, protecting weights
  that mattered, whereas our attribution is provenance — which teacher
  wrote a weight — computed online with no task boundaries and used to
  make permanent writes revertible rather than merely rare.
- **BibTeX:** `liu2026attributionguided`.

## Paper 6 — Titans and Nested Learning

### Titans — Learning to Memorize at Test Time (arXiv:2501.00663, 2024/25)

- **Claim.** A neural long-term memory module (an MLP) trained at test time
  to memorise the context, with the write magnitude controlled by
  "surprise" and a data-dependent forgetting gate; attention is the short-
  term memory, this module the long-term one (Abstract; Sec. 3).
- **Mechanism.** Memory loss `ℓ(ℳ_{t-1}; x_t) = ‖ℳ_{t-1}(k_t) − v_t‖²` (Eq. 12).
  Momentary surprise = `∇ℓ(ℳ_{t-1}; x_t)`, the gradient w.r.t. the memory's
  own parameters; past surprise with momentum (Eqs. 9–10): `S_t = η_t S_{t-1}
  − θ_t ∇ℓ`, `ℳ_t = ℳ_{t-1} + S_t` with data-dependent decay η_t and rate
  θ_t ("similar to gradient descent with momentum, where S_t is the
  momentum element"); forgetting gate `ℳ_t = (1−α_t)ℳ_{t-1} + S_t` (Eqs.
  13–14); read `y_t = ℳ*(q_t)` without update; persistent memory =
  learnable input-independent tokens; chunk-wise parallel training; MAC /
  MAG / MAL variants.
- **Overlap with TriHOPE.** Both use the gradient of a loss w.r.t.
  parameters as the surprise signal and smooth it with a momentum-like
  EMA; Titans' S_t recurrence is SGD-with-momentum, and our `cos(g, m)`
  reads exactly the momentum buffer their S_t is analogous to; a three-
  tier memory story (short / long-term / persistent) maps loosely onto
  R/F/P; their θ_t, η_t, α_t are learned, our thresholds hand-set.
- **Our difference (the "isn't your surprise just Titans'?" answer).**
  What is gated: Titans' surprise scales the write into a *separate learned
  memory module* with its own loss, and high surprise means *write more*;
  TriHOPE gates the permanence of the *ordinary* weights trained on the
  actual task loss, has no auxiliary module, and its high-surprise action
  has the opposite sign — a surprising update is withheld (R) and replayed
  only on recurrence. What the signal is: Titans uses the raw gradient;
  TriHOPE's `S = mean(g²/(v+ε))` uses Adam's v for scale normalisation and
  m only for direction via `cos(g, m)`, a directional-stability statistic
  Titans never computes; no analogue of volatility or recurrence; Titans'
  only gate is an input-conditioned decay, TriHOPE never decays P and
  decides whether to make an update permanent at all. Coordinate-exactness:
  Titans has nothing to hold still; MaskedAdamW leaves closed coordinates
  and their m/v/counters untouched.
- **Baseline implication.** `surprise_gate` (`policy.mode=surprise_only`,
  replay on) covers "route on surprise alone" but is not Titans: theirs is
  a continuous learned scaling of a write that *amplifies* surprising
  writes, ours a threshold that *defers* them; no learned memory MLP, no
  α_t. Wording: "a Titans-style single-signal controller in our setting —
  the ablation that shows S alone does not suffice", never "Titans".
- **One sentence.** Titans (Behrouz et al., 2024) define surprise as the
  gradient of a memory loss with respect to a learned neural-memory module
  and use it, with momentum and a data-dependent forgetting gate, to scale
  writes into that module; TriHOPE borrows the gradient-as-surprise
  intuition but applies it in the opposite direction — a v-normalised,
  m-oriented surprise decides whether an update to the ordinary weights is
  deferred, kept in LoRA, or made permanent, with no separate memory module.
- **BibTeX:** `behrouz2024titans`.

### Nested Learning — The Illusion of Deep Learning Architectures (HOPE) (arXiv:2512.24695, 2025)

- **Claim.** A model is a set of nested optimisation problems, each with its
  own context flow and update frequency; gradient-based optimisers are
  themselves associative memories that compress gradients; a "continuum
  memory system" (CMS) of MLP blocks updated at different frequencies
  generalises short/long-term memory; HOPE = self-modifying Titans + CMS.
- **Mechanism.** Definitions 2–3 (Sec. 3.2): components ordered by update
  frequency, K levels each optimised at its own frequency, each compressing
  its own context flow (tokens for layers, gradients for optimisers).
  Optimisers as associative memory (Sec. 4.2, Eqs. 33–37): momentum is the
  solution of `min_m ⟨m x̂_{ℓ-1}, δ_ℓ⟩`, Adam "the optimal associative memory
  with respect to the element-wise L₂ regression objective" — its second
  moment is the memory that predicts gradient variance. CMS (Sec. 7):
  `y_t = MLP^(f_k)(… MLP^(f_1)(x_t))`, level ℓ updates only every C^(ℓ)
  steps; "higher-frequency neurons … fast adaptation … lower-frequency
  neurons store persistent knowledge"; a standard Transformer is the k = 1
  case. HOPE (Sec. 8): CMS blocks + a self-modifying Titans memory.
- **Overlap with TriHOPE.** The paper TriHOPE's name derives from. NL
  already owns the reading of Adam's m and v as compressed memories of
  gradients and gradient variance — exactly the pair we read; a fast/slow
  hierarchy where slow levels hold persistent knowledge; continual learning
  as multi-frequency memory compression. R/F/P is functionally a three-
  frequency hierarchy (frequency 0 until recurrence / every step / rarely),
  a CMS with k = 3 in spirit. **Never write "optimizer as memory" for
  TriHOPE** — HOPE owns the phrase and the theorem.
- **Our difference.** Level assignment is decided, not scheduled: CMS fixes
  every block's frequency a priori (C^(ℓ) is a hyperparameter); TriHOPE
  assigns each block, at each step, to R/F/P from measured statistics, so a
  block can be slow in one distribution and fast in the next — NL has no
  per-step promotion/demotion rule. Signals are read, not learned: NL uses
  momentum/Adam *as the memory itself*; TriHOPE uses the pre-update
  m_{t-1}, v_{t-1} as diagnostic read-outs, the memory being the plain
  weights + LoRA (the isolating fact: closing a coordinate freezes its
  params *and* its moments, so the moments are the gauge, never the
  store). The merge is a discrete, budgeted permanence event with a count;
  CMS levels never merge. TriHOPE is a controller on a standard Transformer
  + KD pipeline, HOPE a new architecture; we do not claim to be an
  alternative to HOPE.
- **Baseline implication.** None runnable; a faithful CMS baseline would be
  fixed per-block update periods with no signals, still not HOPE. Sentence:
  "we do not compare to HOPE; it is an architecture, TriHOPE a controller;
  the frequency-hierarchy idea is shared."
- **One sentence.** Nested Learning (Behrouz et al., 2025) recasts momentum
  and Adam as associative memories over gradients and proposes a continuum
  memory system whose MLP levels update at fixed, decreasing frequencies so
  that slow levels hold persistent knowledge; TriHOPE keeps the plain
  weights and optimiser but turns the frequency hierarchy into a per-block,
  per-step decision — reading m_{t-1}, v_{t-1} as diagnostics to route each
  update to a deferred, fast-adapter, or permanent tier, rather than
  assigning each block a frequency in advance.
- **BibTeX:** `behrouz2025nested`.

## Paper 7 — Online-LoRA and STABLE

### Online-LoRA — Task-free Online Continual Learning via Low Rank Adaptation (arXiv:2411.05663, WACV 2025)

Authors: Xiwen Wei, Guihong Li, Radu Marculescu (the task document's
"Wei, Kim, et al." is wrong on the second author).

- **Claim.** Task-free online CL for pre-trained ViTs with one trainable
  LoRA pair at a time; distribution shifts are detected from the training-
  loss dynamics, at which point the current LoRA is frozen and merged and a
  fresh one opened; forgetting is limited by a Fisher-style online weight
  regulariser fed from a 4-sample hard buffer (Abstract; Sec. 3).
- **Mechanism.** LoRA only on q/v projections (Sec. 3.2), `Y = (W_init +
  Σ_{t'} B_{t'} A_{t'}) X` (Eq. 1). Trigger (Sec. 3.2, App. C): a sliding
  window over training losses tracks mean and variance; "a peak is
  recognized when the loss window's mean increases by an amount exceeding
  the standard deviation of the window within a single batch"; "a plateau
  is identified when both metrics fall below a predefined threshold" and
  **only if it follows a peak**; per-dataset absolute thresholds (Table 8:
  mean/variance 2.6/0.03 CIFAR-100 … 24.0/1.0 CUB-200); window length not
  stated. At the trigger: freeze the current pair, merge it into the
  pre-trained attention weights, initialise a new trainable pair. Fisher
  penalty on the adapter from the 4 highest-loss samples, λ = 2000
  (Eqs. 4–7). Rank 4, ViT-B/16 / ViT-S/16 / Swin; class- and domain-
  incremental benchmarks.
- **Overlap with TriHOPE.** Task-free, online, LoRA-based, LoRA→base merge
  as the consolidation event — and first (2024) with a simpler recipe.
  "Consolidate at plateau" and "consolidate on sustained stability" can
  look the same from outside (a plateau is a period of small, stable
  gradients); the paper must show they fire at different times.
- **Our difference.** Signal locality: one global scalar with dataset-
  specific absolute thresholds vs per-block decisions from that block's g,
  m_{t-1}, v_{t-1} (different blocks in different tiers at the same step).
  Sign of the trigger: Online-LoRA merges after novelty (peak → plateau);
  TriHOPE merges on the absence of novelty plus repetition (stability ∧
  recurrence) — opposite in the S/C plane and measurable by logging S and
  C̄ at each firing. A third tier (R) with no counterpart; exact masking vs
  a soft quadratic penalty.
- **Baseline implication.** `plateau_trigger` now has the **peak
  precondition** (`consolidation.plateau_require_peak=true`: a plateau
  fires only after the window mean rose by more than its std since the
  last fire — T9) and its merge re-initialises the adapters (fresh pair),
  so it matches Online-LoRA's trigger shape. Remaining gaps to state: our
  plateau test is a relative-improvement test rather than absolute
  mean+variance thresholds; no Fisher regulariser / hard buffer; LoRA on
  every projection TriHOPE uses, not only q/v; all adapters merged at once.
- **One sentence.** Online-LoRA (Wei et al., WACV 2025) is the closest
  task-free LoRA baseline: it trains one adapter at a time, detects a
  distribution shift when a peak in the training-loss window is followed
  by a plateau, and at that point freezes and merges the adapter and opens
  a new one; TriHOPE differs in that its merge is triggered per block by
  sustained directional stability *and* recurrence rather than by a global
  loss plateau, and it has a deferral tier that keeps surprising updates
  out of the adapter altogether.
- **BibTeX:** `wei2025onlinelora`.

### STABLE — Gated Continual Learning for Large Language Models (arXiv:2510.16089, 2025)

Authors: William Hoy, Nurcin Celik. Submitted 17 Oct 2025.

- **Claim.** A gated "continual self-editing" framework: each sequential
  LoRA edit is a candidate merge that must satisfy a forgetting budget on
  previously edited anchors, measured by EM drop, bits increase, or KL
  drift; otherwise the adapter is rescaled by binary search or rejected.
- **Mechanism.** Base θ_base; each edit trains a LoRA (rank 32, α 64, 10
  epochs, Qwen-2.5-7B), merges if accepted, 8 sequential edits (Sec. 4,
  App. B). Anchors = previously edited datapoints (growing). Gate (Sec.
  3.1.1, 3.2, Alg. 1): accept iff `f ≤ ε` with `f_EM = max(0, EM_base −
  EM_adapter)`, `f_bits = max(0, bits_adapter − bits_base)`, or `f_KL` (a
  per-token log-ratio on the anchors, bits/token); budgets ε = 7 % (EM),
  0.08 bits/token, 0.7 bits/token. On rejection a binary search over the
  scaling α ∈ [0.1, 1] finds the largest passing `f(α·W_LoRA) ≤ ε` (5
  evaluations); else rejected. EM gating gave the best cumulative gain
  (+0.397 over 8 steps). The gate is applied to the whole adapter; no
  gradients, moments or optimizer state are read.
- **Overlap with TriHOPE.** The cleanest prior statement of our reframe:
  *permanence (the merge) is a gated, budgeted decision*, in an LLM
  setting at 7B, with an outcome-level probe that directly measures
  forgetting — which our optimizer statistics only proxy. Both accept /
  reject a LoRA→base merge; both keep base + adapters; both sequential.
- **Our difference.** STABLE's gate is extrinsic (forward passes over an
  anchor set, up to 5 evaluations per edit, base vs adapted outputs); ours
  is intrinsic and free (g, m_{t-1}, v_{t-1} already in memory, every step,
  per block, no probe data). Granularity and timing: STABLE gates one whole
  adapter after 10 epochs; TriHOPE gates per block per step and decides
  *before* training whether an update even enters the adapter (R). On
  failure STABLE rescales or rejects; TriHOPE waits with exact optimizer
  state on un-merged coordinates. No distillation, no multi-teacher, no
  boundary-free stream in STABLE.
- **Baseline implication.** **No baseline exists in the repo** — recorded
  as a follow-up in `docs/STATUS.md`. A faithful `stable_gate` would need:
  anchors = rows already consolidated into P (growing); at each candidate
  merge compute `f_KL` (per-token log-ratio of adapted vs base on the
  anchors) or `f_EM` against ε; binary search over α ∈ [0.1, 1] with ≤ 5
  evaluations, merge α·ΔW or reject; no use of g/m/v; matched permanent-
  write count; report the forward-pass cost per merge. Without it the
  "budgeted permanence" framing is exposed to "does a cheap intrinsic gate
  lose much against a probe?".
- **One sentence.** STABLE (Hoy and Celik, 2025) already treats each
  LoRA→base merge as a candidate that must pass a forgetting budget —
  measured by exact-match drop, bits increase, or KL drift on previously
  edited anchors — and rescales or rejects it; TriHOPE shares the view of
  permanence as a gated decision but gates from the optimiser's own
  gradient and moment statistics, per block and per step, with no probe
  set or extra forward passes, and adds a deferral tier that STABLE lacks.
- **BibTeX:** `hoy2025stable`.

## Paper 8 — Merge before Forget, Sparse memory finetuning

### SLAO — Merge before Forget: A Single LoRA Continual Learning via Continual Merging (arXiv:2512.23017, 2025)

- **Claim.** One LoRA suffices for continual learning: sequentially merge
  each task's LoRA into a single running LoRA with a time-aware scaling and
  initialise each new task orthogonally from the previous one; constant
  memory in the number of tasks (Abstract; Sec. 1).
- **Mechanism.** `W0` frozen; per task: initialise `A_ft,i` from the QR
  decomposition of the previous task's `A_ft,i-1ᵀ` (Eq. 13), `B_ft,i^(0) =
  B_ft,i-1`; fine-tune; merge at the **task boundary** (Alg. 1): `A_merge =
  A_ft,i` (replace), `B_merge^i = B_merge^(i-1) + λ(i)(B_ft,i − B_merge^(i-1))`
  with `λ(i) = 1/√i` (Eq. 14); motivated by an NTK forgetting bound
  (Lemma 1, Thm. 1). Exactly one `{B_merge, A_merge}` pair kept; **LoRA is
  never folded into W0**; trigger purely the task boundary; SGD-style
  updates; no signals, retrieval, replay or distillation.
- **Overlap with TriHOPE.** **Two-tier** (frozen base + one LoRA shared over
  time — the same single-adapter-over-time arrangement as our F tier) with
  an explicit merge rule and constant memory; their Lemma 1 bounds
  forgetting by the LoRA-delta drift, the quantity our merge waits to see
  stable.
- **Our difference.** Merge target: LoRA→LoRA, never W0 (permanence is not
  a state) vs our LoRA→base. Trigger: known task boundaries with a fixed
  `1/√i` schedule vs a per-block signal condition with no boundary
  knowledge that can decline to merge for a whole phase. No R tier, no
  optimizer-state gating, no MaskedAdamW, anonymous merges vs teacher-
  tagged, checkpointed, selectively revertible ones; single-task-sequence
  fine-tuning vs online multi-teacher distillation.
- **Baseline implication.** `lora_only` is essentially SeqLoRA (which SLAO
  beats) — covers the tier structure, not SLAO's contribution;
  `plateau_trigger` covers "merge on a scalar trigger" into base, not a
  boundary-triggered LoRA→LoRA merge with `1/√i` damping. Not needed to
  defend the permanence claim (SLAO has no permanent tier); an `slao_style`
  spec (merge at phase boundaries with `λ = 1/√i` into the shared LoRA, QR
  re-init) is cheap if reviewers ask.
- **One sentence.** SLAO (Qiao & Mahdavi, 2025) keeps a single LoRA by
  merging each task's adapter into it at task boundaries with a `1/√i`
  time-aware scaling and orthogonal re-initialisation, but the merge is
  boundary-triggered and never reaches the base weights, so there is no
  notion of a per-block decision that an update has earned permanence.
- **BibTeX:** `qiao2025mergebeforeforget`.

### Sparse memory finetuning — Continual Learning via Sparse Memory Finetuning (arXiv:2510.15103, 2025)

- **Claim.** Updating only the memory-layer slots that new data activates
  far more than pretraining data did lets a model learn new facts with
  little forgetting: 11 % NaturalQuestions F1 drop vs 89 % for full fine-
  tuning and 71 % for LoRA (Abstract; Sec. 5.1).
- **Mechanism.** A 1.3B Transformer with the FFN of layer 12/22 replaced by
  a product-key memory layer (1M slots, k = 32 accesses, 4 heads). Per
  batch every slot gets a TF-IDF score `c(i)/Σ_j c(j) · log((|B|+1)/(Σ_b
  1[c_b(i)>0]+1))` against 1000 background pretraining batches; the top-t
  slots (t = 500 / 10 000) are chosen per batch; only their **values** are
  finetuned (stop-gradient elsewhere); keys, non-selected values and all
  non-memory parameters frozen (Sec. 4). SGD for the sparse method; the
  paper warns that "adaptive per-parameter step sizes, weight decay, and
  momentum can interact with sparsity in unexpected ways" (Sec. 5). No
  promotion, consolidation, retrieval, replay or second tier.
- **Overlap with TriHOPE.** Also **two-tier** (per-batch-selected written
  set vs frozen rest) — our F/not-F split at coordinate level, with a stop-
  gradient mask playing the role of MaskedAdamW. Their TF-IDF score is a
  *recurrence-vs-background* statistic — the same family as our repetition
  signal and, like the bucket counter, data-derived rather than optimizer-
  derived. Their warning about adaptive optimizers and sparse updates is a
  direct warning against our design.
- **Our difference.** Architecture-agnostic (no memory layers; F = Top-K
  LoRA components on a standard Transformer, P = base); three states with
  promotion rules (R→F on recurrence, F→P on stability ∧ recurrence) vs
  write/frozen; selection read from the gradient and pre-update moments per
  block per step, no background corpus; multi-teacher, teacher-tagged,
  checkpointed merges.
- **Baseline implication.** Not runnable without a memory-layer student.
  `molf_style` covers "signal-selected sparse write, rest frozen",
  `lora_only` the always-on adapter; coverage of the specific method: none
  — say so. Cite the 11 % vs 71 % number as evidence that write-sparsity is
  the lever, and answer their optimizer warning: MaskedAdamW keeps closed
  coordinates and their moments untouched, which is the mechanism that
  avoids the interaction they observed.
- **One sentence.** Sparse memory finetuning (Lin et al., 2025) updates
  only the top-t memory-layer values whose TF-IDF access score on the
  current batch exceeds their pretraining usage and freezes everything
  else, a two-tier write/frozen split that has no deferred tier and no rule
  for when a sparse write should become permanent.
- **BibTeX:** `lin2025sparsememoryfinetuning`.

## Paper 9 — Poisoned teachers

### ABD — Revisiting Data-Free Knowledge Distillation with Poisoned Teachers (arXiv:2306.02368, ICML 2023)

- **Claim.** A backdoored teacher transfers its backdoor to the student
  through data-free KD far more readily than through KD on clean data;
  Anti-Backdoor Data-Free KD (ABD) suppresses the transfer at a small
  clean-accuracy cost.
- **Mechanism.** Threat model (Sec. 2): the attacker releases a backdoored
  teacher; the defender has only the teacher, no data, and does not know
  whether it is poisoned. DFKD is vulnerable because synthetic-input
  generation maximises teacher–student disagreement and drifts toward
  trigger-like inputs, and the teacher's soft labels carry the backdoor on
  OOD inputs (Sec. 3); clean-data KD transfers at 0.7–33 % ASR, DFKD at up
  to 100 %. Defence (Sec. 4): Shuffling Vaccine (score inputs by
  `log D_KL(T̃(x)‖T(x))` against a channel-shuffled teacher and suppress
  suspicious samples) and Self-Retrospection (a bi-level unlearning step in
  the last epochs). Results: e.g. BadNets-grid 96.9 % → 4.3 % ASR at −5 to
  −8 pp clean accuracy (Table 1, 3). Single teacher, vision, no provenance.
- **Overlap with TriHOPE.** Small: the shared premise that a teacher is an
  untrusted artifact whose influence enters through the distillation loss.
  Our threat (corrupted cached outputs for one phase) is not a backdoor;
  ABD is prevention, we offer attribution and rollback — do not claim to
  "defend against poisoned teachers" in their sense.
- **Our difference.** Multi-teacher with per-teacher tagging; permanence is
  gated (a corrupted teacher's updates must pass R→F→P); merges are
  checkpointed so permanent writes can be located and reverted selectively;
  cached traces, not data-free synthesis (their "bad synthetic input supply"
  does not apply; "bad supervision" does — that is what tagging is for).
- **Baseline implication.** No run; cite as the recognised-threat reference
  in E5's motivation, stating that our threat is output corruption and our
  remedy provenance plus rollback.
- **One sentence.** Hong et al. (2023) show that a backdoored teacher's
  behaviour transfers through (data-free) distillation at > 90 % attack
  success for most triggers and propose ABD to suppress it at training
  time, establishing untrusted teachers as a real threat; we do not attempt
  prevention but make every permanent write attributable to a teacher and
  revertible.
- **BibTeX:** `hong2023poisonedteachers`.

## Paper 10 — HippoRAG 2

### HippoRAG 2 — From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (arXiv:2502.14802, ICML 2025)

- **Claim.** Retrieval over a knowledge-graph-plus-passage index with
  Personalized PageRank is a practical form of continual learning that
  needs no parameter updates and beats standard RAG on factual, sense-
  making and associative memory.
- **Mechanism.** Continual learning "generally fall[s] into three
  categories: continual fine-tuning, model editing, and RAG"; RAG "retrieves
  relevant external information at inference time … without altering an
  LLM's parametric representation" (Sec. 1–2). Offline: OpenIE triples,
  phrase/relation/synonym/passage nodes (Sec. 3.1–3.2); online: triple
  matching, an LLM recognition-memory filter, PPR, top-5 passages (Sec.
  3.3–3.5). Avg F1 59.8 vs NV-Embed-v2 57.0 (Table 2); robust to corpus
  expansion (Sec. 6.3). **Nothing is ever written into the LLM's
  parameters, and no rule is stated for when retrieved knowledge would
  become parametric.**
- **Overlap with TriHOPE.** Our R tier is a store, and we too argue that
  not every observation deserves a parameter update; their retriever is far
  more sophisticated — do not compare on retrieval quality.
- **Our difference.** External forever vs deferred then promoted: in
  HippoRAG 2 knowledge stays in the index and the LLM is a fixed reader; in
  TriHOPE a parked row is replayed into LoRA when its bucket recurs and an
  F update that shows stability ∧ recurrence is merged into base — an
  explicit, signal-based rule for when non-parametric memory becomes
  parametric, which HippoRAG 2 explicitly does not have.
- **Baseline implication.** No run; the in-framework proxy for "external
  forever" is `trihope_r_terminal` (`replay_on_hit=false`) in
  `r_tier_small.yaml`, and `no_retrieval` is the opposite corner.
- **One sentence.** HippoRAG 2 (Gutiérrez et al., 2025) casts retrieval as
  non-parametric continual learning and keeps all new knowledge external
  with the LLM as a fixed reader, offering no criterion for when a
  retrieved fact should become parametric; TriHOPE's R tier is instead a
  deferral that promotes a row into the adapter on recurrence and into the
  base weights on sustained stability.
- **BibTeX:** `gutierrez2025hipporag2`.

---

## Cross-cutting cautions for the paper text

1. Every rival is two-store and decides *where* once (MoLF, SLAO, sparse
   memory, Gradient Routing, Online-LoRA, STABLE); the defensible novelty
   is the third state (deferral in R, promotion to P), the optimizer-state
   gating, exact masking, and provenance — not "sparse writes" or "a single
   shared adapter".
2. Sparse memory finetuning's and OGP's warnings about adaptive optimizers
   under attenuation/sparsity should be cited and answered with the exact
   mask (closed coordinates keep their moments).
3. Terminology: R is a *deferral* tier (not retrieval in the ReGrad sense);
   attribution is *provenance* (not importance, as in AGFT); never
   "optimizer as memory" (Nested Learning).
4. STABLE is the closest statement of "permanence as a budgeted decision";
   the missing `stable_gate` baseline is the one reviewers are most likely
   to ask for (design in Paper 7).
5. OGP is withdrawn; cite v1 for Proposition 1 only.

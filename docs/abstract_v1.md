# ICLR 2027 — title + abstract v1 (for the Sep 18 abstract registration)

Written against `docs/TriHOPE_ICLR2027_Reframe.md` (§2 claims, §7 writing changes)
and `docs/STATUS.md` (E1–E5 results still pending the GPU campaign).
The abstract states no results: the campaign lands Sep 13–16, registration is
due Sep 18, and nothing has been run on the real stream yet. Results wording
for the Sep 25 paper version is kept separately below.

---

## Recommended title

**How Permanent Should This Update Be? Optimizer-State Evidence for Deferred,
Fast, and Permanent Writes in Continual Distillation**

### Alternates

1. *Learning Rates Say How Much, Not How Permanently: Budgeted Write Routing
   from Pre-Update Adam State* — sharpest hook, weakest keyword coverage.
2. *Which Teacher Wrote This? Attributable, Evidence-Gated Routing of
   Distillation Updates Across Deferred, Fast, and Permanent Memory* — the
   reframe doc's working title; lead with this only if E5 becomes the headline.
3. *Permanence as a Budgeted Decision: Routing Online Updates Across Three
   Memory Tiers from Adam's Moments* — most conservative/descriptive.

### Naming note

Keep the method name out of the title, or rename it. **TriHOPE** advertises
descent from Nested Learning / HOPE (Behrouz et al.), which is exactly the
prior work that owns "optimizer as memory" — the name invites the reviewer to
read the paper as a HOPE variant. If a name is wanted, `TRIAGE`
(TRI-store Adam-Gated Evidence) says what the controller does without the
inheritance.

---

## Abstract v1 (~225 words, no results, no em dashes)

Optimizers decide how large each update is; nothing decides how permanent it
should be. In a task-free stream distilled from several teachers of uneven
quality, an update can be worth taking now and wrong to keep forever. It may be
a one-off observation, a pattern that has not settled, or the output of a
teacher that later turns out to be corrupt. Ordinary Adam writes all of them
into shared weights, entangled with everything else. We treat permanence as a
decision and read the evidence for it from state the optimizer already keeps.
A block-level controller reads the pre-update gradient together with Adam's
first and second moments, with no extra gradients, probes, or forward passes,
and estimates surprise, directional agreement, volatility, and recurrence. Each
update is then routed to one of three commitment levels: deferred to a
non-parametric store that writes no weights and is replayed only if the
observation returns, written tentatively to sparse LoRA fast weights, or merged
into base weights under sustained stability and recurrence. A masked AdamW
makes the boundary exact: a closed coordinate receives no gradient step, moment
update, weight decay, or bias-correction aging. The run therefore becomes a
ledger, in which the total parameter change decomposes into logged writes, each
tagged with the teacher and the signal values that authorised it. We study this
policy on a 6000-step, seven-phase stream with a 0.6B student and four
teachers, comparing it against loss-plateau merging, surprise-only gating, an
Adam-score two-tier policy, and a share-matched random control at a matched
permanent-write budget, and measuring how much of a corrupted teacher's
influence reaches base weights and how much of it selective rollback recovers.
This is a controlled mechanism study, not a claim of a stronger student.

---

## Adding results for the Sep 25 paper version

The registered abstract above states no findings. For the full paper, add one
sentence after the study description; the wording depends on the Sep 13
go/no-go (reframe section 9):

- **Go:** "Reading the optimizer traces a better forgetting/plasticity frontier
  than [baselines] at matched permanent writes, and separates clearly from the
  share-matched control."
- **Conditional go:** "Reading the optimizer matches the strongest trigger
  baseline on the frontier at no additional cost, and separates clearly from
  the share-matched control." Then lead with attribution and rollback, and
  switch to title #11.
- Follow with the E5 numbers: the containment fraction (share of the corrupted
  teacher's actions ending in R or F rather than P) and the selective-rollback
  versus full-restore recovery gap.

## What the abstract deliberately does not say

- No "optimizer as memory" (HOPE), no "surprise" as a novel signal (Titans),
  no "function-preserving merge" as a contribution (it is the LoRA identity).
- No student-quality or benchmark claim — §1 of the reframe kills it.
- No teacher-selection claim; teacher routing is setup, not contribution.

---

## Title bank (v2)

Constraints every candidate respects: no "memory as optimizer" / HOPE echo, no
student-quality claim, no result asserted before E1 lands, and enough bidding
keywords (continual, forgetting, optimizer state, Adam, LoRA, distillation)
for the right reviewers to bid.

### A. Permanence-as-a-decision (the reframe's own thesis)

1. When Should an Update Become Permanent? Optimizer-State Routing Across
   Deferred, Fast, and Permanent Memory
2. How Permanent Should This Update Be? Reading Adam's Moments to Route Writes
   Across Three Memory Tiers
3. Permanence as a Budgeted Decision in Task-Free Continual Distillation
4. Not Every Update Deserves the Base Weights: Evidence-Gated Write Routing
   from Pre-Update Adam State
5. Step Size Without Commitment: Deciding How Permanently to Write Each Update

### B. Mechanism-forward, descriptive (safest, best keyword coverage)

6. Optimizer-State Write Routing: Deferred, Fast, and Permanent Memory for
   Task-Free Continual Distillation
7. Routing Online Updates to Retrieval, LoRA, and Base Weights with Pre-Update
   Adam Statistics
8. Exact Write Gating from Adam's Moments for Task-Free Continual Learning
9. A Three-Tier Write Policy for Continual Distillation, Driven by Pre-Update
   Optimizer State
10. Block-Level Write Routing from Adam Moments for the Forgetting–Plasticity
    Trade-off

### C. Attribution / rollback lead (use if E1 ties and E5 carries the paper)

11. Which Teacher Wrote This? Attributable and Revertible Permanent Writes in
    Continual Distillation
12. A Ledger for Learning: Attributable Permanent Writes in Multi-Teacher
    Continual Distillation
13. Undoing a Bad Teacher: Attribution and Selective Rollback of Permanent
    Weight Writes
14. Containing a Corrupted Teacher: Evidence-Gated Permanence in Multi-Teacher
    Distillation
15. Weight Provenance: Optimizer-Gated Writes That Can Be Traced and Reverted

### D. If the method keeps a name (TRIAGE = TRI-store Adam-Gated Evidence)

16. TRIAGE: Evidence-Gated Routing of Online Updates Across Deferred, Fast, and
    Permanent Memory
17. TRIAGE: Deciding How Permanently to Write Each Update from Pre-Update Adam
    State

### Picks

- **Default (survives either E1 outcome):** #1
- **Most conservative / best for reviewer bidding:** #6
- **Conditional-go branch (E5 headline):** #11

### Avoid

"Towards …", "Rethinking …", "A Closer Look at …", "Beyond …", "Is X All You
Need?" — all read as filler in 2026 review. Also avoid any title asserting an
improvement ("improves", "outperforms", "better") until E1 is in.

---

## OpenReview submission fields (ICLR 2027)

**Keywords** (comma separated, ordered broad to specific so reviewer matching
lands in the continual-learning pool first):

```
continual learning, catastrophic forgetting, knowledge distillation, optimizer state, Adam, LoRA, parameter-efficient fine-tuning, online learning, model merging, training dynamics
```

Swaps if a narrower audience is wanted: `task-free continual learning`,
`memory consolidation`, `gradient routing`, `streaming data`,
`training data attribution`, `plasticity-stability trade-off`.
Drop `Adam` only if the list feels crowded; it is the single most specific
matching term for the optimizer-state reviewers.

**Primary Area:** `transfer learning, meta learning, and lifelong learning`.
The evaluation is forgetting and retention over a task-free stream, so
lifelong learning is the closest fit. `optimization` is the second choice but
routes to convergence-theory reviewers who will want guarantees the paper does
not claim.

**TL;DR:** Adam's own moments say whether an update should be deferred, held in
LoRA, or written permanently into the base weights, and enforcing that split
exactly at the optimizer makes every permanent write teacher-attributed and
revertible.

**AI Assistance:** disclose honestly. This work used an assistant for writing
and drafting, so tick "to aid or polish writing", "to draft sections of the
paper", and whichever of "retrieval and discovery" / "research ideation or
execution" applies; a matching disclosure section is mandatory in the PDF.

**Deadline note:** the OpenReview form shows Sep 19, 2026 5:59 PM (UTC), which
is Sep 18 AoE, matching the reframe timeline. `Reciprocal Reviewing Author` and
the exemption fields are frozen at the abstract deadline; title, keywords,
TL;DR and abstract stay editable until the full paper deadline.

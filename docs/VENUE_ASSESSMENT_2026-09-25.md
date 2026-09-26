# Venue assessment against DP1–DP4 (2026-09-25)

Live, KD active. Baseline: E1 live `trihope` **0.0805 ± 0.0101** (n=3).
Metric: mean over {general, code, math, medical} of `worst_retention_delta`, lower better.

| Decision point | Arm | Result | vs baseline | Read |
|---|---|---:|---:|---|
| **DP4** "tests the paper's headline claim" | `random_commit` ×3 | 0.0841 ± 0.0130 | +0.0036 | **null** |
| **DP1** "decides the paper's framing" | `frozen_blocks` ×3 | 0.0841 ± 0.0152 | +0.0036 | **null** |
| DP2 drift (rule: inside 0.0838 ± 0.012) | `trihope_sentinel` ×3 | 0.0838 ± 0.0169 | +0.0032 | **pass** |
| DP3 competitor | `surprise_gate_s4` ×3 | 0.0743 ± 0.0109 | −0.0062 | ties/ahead |
| DP3 competitor | `molf_style_a0p7` ×3 | 0.1024 ± 0.0177 | +0.0218 | behind |
| DP3 ablation | `no_cosine` ×3 | 0.1649 ± 0.0180 | +0.0844 | signal matters |
| Gate dose-response | `pgate_c0p35` | 0.0806 | +0.0001 | null |

## Why DP4 is decisive

`random_commit` keeps TriHOPE's defer/replay machinery and **reassigns commits at random** at
TriHOPE's own per-phase commit share — by design it isolates commit *selection* from replay volume.
It lands inside one standard deviation of TriHOPE. So the controller's learned commit decisions are
not measurably better than random ones; whatever is working is the defer/replay scaffold, not the
routing intelligence the paper claims.

`frozen_blocks` says the same thing from the other side: train only the shared (unrouted) params,
with no routing and provably untouched routed weights, and retention is statistically identical.

`no_cosine` degrading (+0.084) is consistent with this rather than against it — the cosine signal
changes *how much* gets written, and write volume drives forgetting (the +0.74 replay-count
correlation in the cache-mode grid). It does not show that commit selection is informative.

## What is solid

- **Theorem 1 verified at scale**: `frozen_blocks` routed weights bit-identical across 3 seeds —
  `sq_norm_delta 0.0`, `changed_coords 0 / 440,401,920`, `sum_frob_BA 0.0`.
- **Reproducibility**: DP2 passes exactly; the sentinel reproduces E1 TriHOPE.
- **Catastrophic forgetting is real and prevented**: `full_ft` 0.8725, `molf_style` 0.3026,
  `lora_only` 0.1828 vs 0.0805.
- **Pre-registered decision points, honoured** — including the ones that came back null.

## Venue

**Oral (~top 1%) and spotlight (~top 5%): not supportable.** Both need a central claim that holds.
Two pre-registered tests of the central mechanism returned null, and a reviewer running exactly the
comparison the authors designed would find it.

**Main-conference acceptance as a positive-result paper: not supportable either, as framed.**

**Credible as a rigorous negative / diagnostic paper.** The contribution becomes: gradient-signal
routing for continual distillation does not beat random commit selection once replay volume is held
fixed, shown with pre-registered decision rules, a verified invariance theorem, n=3 seeds, and a
mechanism (write volume, not write choice). That is a real and useful result. Workshop or
main-conference as a negative result, not a spotlight.

## Gaps a reviewer will raise regardless

- One scale only: Qwen3-0.6B student, `stream_small`. Nothing shows the ordering holds larger.
- n=3 at best; much of the ablation grid is n=1 with a borrowed noise floor.
- Live vs cache differ in **both** KD activity and thresholds, so the two result sets are not paired.
- Cut this batch: `molf_style_a0p5`, `moments_optimizer`, `no_surprise`, `topm_2`, `topk_25`,
  `topm_all`, 3 of 4 `pgate`, and `periodic_merge` (the merge baseline that actually fires).
- The design-faithful variant and the frozen-shared-param TriHOPE arm were never run (already
  listed as limitations).

---

## Addendum (2026-09-25): the acquisition check closes the rescue

Codex's independent read raised the strongest available rescue: forgetting alone
rewards a model that learns nothing, so `frozen_blocks` might look good only
because it acquires little. Separating acquisition from retention is standard
practice (RWalk, arXiv:1801.10112). Measured from `retention.own_phase_loss`
(each phase's own domain; lower = learned more):

| arm | n | own-phase loss | forgetting |
|---|---:|---:|---:|
| `lora_only` | 3 | **1.1974 ± 0.008** | 0.1828 |
| `trihope` | 3 | 1.2490 ± 0.007 | 0.0805 |
| `trihope_sentinel` | 3 | 1.2458 ± 0.003 | 0.0838 |
| `random_commit` | 3 | 1.2523 ± 0.007 | 0.0841 |
| `frozen_blocks` | 3 | 1.2577 ± 0.003 | 0.0841 |
| `surprise_gate_s4` | 2 | 1.2603 ± 0.012 | 0.0743 |
| `no_cosine` | 2 | 1.2791 ± 0.007 | 0.1649 |
| `gold_ce` | 3 | 1.5141 ± 0.003 | 0.0596 |

`trihope` 1.2490 ± 0.007 vs `random_commit` 1.2523 ± 0.007: indistinguishable on
acquisition too. `frozen_blocks` at 1.2577 is within ~1 sd, so it is **not**
buying retention by refusing to learn. The rescue fails on its own terms — as
Codex framed it, "if frozen blocks and surprise-only also match acquisition, the
hidden-mechanism rescue has little support." They match.

What the two axes together do show is a real stability/plasticity trade-off:
`lora_only` acquires most (1.1974) and forgets most (0.1828); `gold_ce` retains
best (0.0596) and acquires least (1.5141). `trihope` sits mid-front — but so does
`surprise_gate_s4`, a single threshold, at equal or better forgetting. The full
controller's complexity is unjustified by either axis.

## The one experiment that would still settle it

Codex's replay-yoked 2×2, which neither of us has run: full controller vs
cosine-blind routing, crossed with two externally imposed replay budgets, with
replay examples, timing, token counts and KD weighting held identical inside each
budget, and the cosine-blind arm preserving the full arm's per-step R/F/P quotas
while randomising only cosine's allocation among comparable modules. That
separates "cosine carries information" from "cosine throttles write volume" —
the confound the current `no_cosine` arm cannot resolve, since removing the signal
also removes the throttle.

This needs a new policy branch plus 4 arms x 3 seeds = 12 runs (~14 h). It is the
honest route to a positive claim, and it may still come back null.

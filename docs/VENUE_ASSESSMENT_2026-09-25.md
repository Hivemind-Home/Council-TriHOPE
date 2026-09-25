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

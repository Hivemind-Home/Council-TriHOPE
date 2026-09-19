# TriHOPE — consolidated results, all experiments

Generated 2026-09-19 from the verified box mirror (142/142 run summaries, every arm `state=done, exit_code=0`).

Metric: **mean forgetting** = mean over {general, code, math, medical} of `worst_retention_delta`. **Lower is better.** n=3 seeds where a `±` is shown, n=1 (seed 1337) otherwise.

## E1 — baselines (n=3)

| arm | mean forgetting | vs trihope |
|---|---:|---:|
| `gold_ce` | 0.0529 ± 0.004 | -0.0603 |
| `no_retrieval` | 0.0706 ± 0.009 | -0.0426 |
| `no_consolidation` | 0.1104 ± 0.010 | -0.0029 |
| `trihope` | 0.1132 ± 0.009 | — baseline |
| `trihope_no_hash` | 0.1331 ± 0.046 | +0.0199 |
| `surprise_gate` | 0.1436 ± 0.124 | +0.0304 |
| `plateau_trigger` | 0.1823 ± 0.063 | +0.0691 |
| `lora_only` | 0.1841 ± 0.065 | +0.0709 |
| `random_routing` | 0.3011 ± 0.072 | +0.1879 |
| `molf_style` | 0.4587 ± 0.122 | +0.3455 |
| `full_ft` | 3.0951 ± 2.102 | +2.9818 |

## E4 — ablation grid, replay-ON, in-grid baseline (n=1)

| arm | mean forgetting | vs trihope |
|---|---:|---:|
| `no_repetition` | 0.0514 | -0.0448 |
| `topm_2` | 0.0692 | -0.0270 |
| `topk_25` | 0.0755 | -0.0207 |
| `consolidation_strict` | 0.0909 | -0.0053 |
| `trihope` | 0.0961 | — baseline |
| `no_volatility` | 0.0975 | +0.0014 |
| `topk_100` | 0.1033 | +0.0071 |
| `no_teacher_conf` | 0.1063 | +0.0102 |
| `stability_instant` | 0.1195 | +0.0234 |
| `topm_all` | 0.1392 | +0.0430 |
| `no_surprise` | 0.1897 | +0.0936 |
| `no_cosine` | 0.2078 | +0.1117 |
| `moments_optimizer` | 0.2498 | +0.1536 |

## E5 — bad-teacher robustness (n=3)

| arm | mean forgetting | vs trihope |
|---|---:|---:|
| `molf_snr` | 0.0393 ± 0.006 | -0.0693 |
| `trihope_forced` | 0.0878 ± 0.012 | -0.0207 |
| `trihope` | 0.1086 ± 0.013 | — baseline |
| `trihope_nogate` | 0.1086 ± 0.014 | +0.0000 |
| `trihope_lowconf` | 0.1100 ± 0.017 | +0.0015 |
| `lora_only` | 0.1736 ± 0.061 | +0.0650 |
| `molf_style` | 0.6876 ± 0.552 | +0.5790 |
| `gradient_routing` | 1.2211 ± 0.735 | +1.1125 |
| `full_ft` | 2.5914 ± 2.002 | +2.4829 |
## Assessment

### What the data supports, strongly

TriHOPE beats every naive baseline by margins far outside the error bars, at n=3:
`full_ft` (3.10 vs 0.113 in E1), `molf_style` (0.459), `random_routing` (0.301),
`lora_only` (0.184), and in the bad-teacher setting `gradient_routing` (1.22) and
`molf_style` (0.688). Catastrophic forgetting is real in this setup and the method
prevents it. That result is solid and reproducible.

The ablation grid also establishes a clean positive mechanism claim: **surprise and
sustained cosine are load-bearing.** Removing either costs 16-19x the seed-noise floor
(`no_surprise` +0.094, `no_cosine` +0.112), and replacing the controller with raw Adam
moments is worse still (`moments_optimizer` +0.154, 26x noise).

### What blocks an "our method wins" paper

Three independent results point the same way, and none is a fluke of one seed:

1. **A fairly configured MoLF baseline beats the method outright.** E5, n=3:
   `molf_snr` 0.0393 ± 0.006 vs `trihope` 0.1086 ± 0.013 — 2.8x better, error bars
   nowhere near overlapping. The `molf_style` baseline that TriHOPE does beat was
   configured with `top_k_fraction=1.0` (write everything) against trihope's 0.5; the
   fair variant dominates.
2. **Several of the method's own components are inert or harmful.** `trihope_nogate`
   ties trihope to four decimals (gate does nothing). `no_consolidation` ties within
   noise. `no_retrieval` *beats* it by 0.043 in E1, and `no_repetition` by 0.045 in E4.
3. **The shipped configuration over-writes.** Both budget reductions tested
   (`topm_2` -0.027, `topk_25` -0.021) improve retention.

So the honest summary is: the tri-store routing idea works against naive training, but
the *specific system as shipped* is beaten by a simpler published method and by four of
its own ablations. Submitting it as a win would not survive review — a reviewer running
the fair MoLF comparison would find what E5 already shows.

### The paper this data can support

Reframed as a diagnostic study, the same runs are publishable and honest: *which
gradient signals actually matter for routing in continual distillation.* The ablation
grid is the contribution — surprise and cosine carry the effect, repetition/retrieval/
consolidation/confidence-gating do not, and an SNR rule outperforms the full controller.
That is a useful negative result with a clear mechanism, and it is well-supported at the
sample sizes available.

### Caveats that still apply

- E4 is **n=1 per arm**; its noise floor is borrowed from one other arm's seed spread.
  Deltas in the 2-4x band (`topk_25`, `stability_instant`) need n=3 before publication.
- Reported error bars understate uncertainty (see `docs/STATUS.md` §2b on nondeterminism).
- All results are at the `stream_small` scale with a Qwen3-0.6B student. Nothing here
  establishes that the ordering holds at larger scale.

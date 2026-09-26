# Pre-registration: secondary analyses (written BEFORE computing them)

Codex named four measurements absent from our campaign that could carry a positive
claim. Two are computable from data already on disk. They are declared here, with
their decision rules, **before** the numbers are looked at, because the honest
failure mode is obvious: search enough metrics and one will look like a win.

**The primary metric does not change.** Mean `worst_retention_delta` over
{general, code, math, medical} stays primary, and DP1–DP4 stand as already
reported (DP4 and DP1 null). Everything below is secondary and will be labelled
as such in any write-up, whatever it shows.

## S1 — Matched-acquisition efficiency

Cost to reach equal retained capability: `wall_clock_s`, teacher calls
(`ledger_totals.teachers.*`), replay rows (`ledger_totals.replay_count`) and
committed coordinates (`coords_opened`, `merged_coords`), each read against
own-phase loss so that arms which simply learn less are not credited.

- **Supports a positive claim** if `trihope` reaches the same own-phase loss as a
  competitor at materially lower cost on at least one axis, with the cost gap
  outside the seed spread.
- **Null** if cost differences are within seed spread, or if lower cost is
  explained by lower acquisition.

## S2 — Functional interference through shared parameters

`frozen_blocks` trains only the shared (unrouted) parameters and its routed
weights are verified bit-identical (0 of 440,401,920 coords). Its forgetting
therefore measures how much of our forgetting is transmitted by shared parameters
alone. This directly probes the gap Codex identified in our Theorem 1 claim:
parameter immutability is not functional invariance.

- **Supports a positive claim** if `frozen_blocks` forgetting is materially BELOW
  `trihope`, i.e. routing suppresses a real shared-parameter interference channel.
- **Refutes the value of routing** if `frozen_blocks` forgetting matches or beats
  `trihope`: the protection our routing provides would then be irrelevant, because
  forgetting does not travel through the routed weights it protects.
- Either way this is a mechanism result, not a performance result.

## Not computable without new GPU time

S3 teacher-conflict robustness under controlled disagreement, and S4
recovery/transfer including within-phase collapses hidden by checkpoint-only
evaluation (stability gaps, arXiv:2205.13452). Both are recorded as future work,
not run.

---

# Results (computed after the rules above were committed)

## S1 — matched-acquisition efficiency: NULL, trending negative

| arm | acq (own-phase loss) | forgetting | coords opened (M) | replay rows |
|---|---:|---:|---:|---:|
| `lora_only` | 1.1974 | 0.1828 | 60,555 | 0 |
| `trihope` | 1.2490 | 0.0805 | **686** | 1672 |
| `random_commit` | 1.2523 | 0.0841 | 678 | 1634 |
| `frozen_blocks` | 1.2577 | 0.0841 | **0** | 0 |
| `surprise_gate_s4` | 1.2603 | 0.0743 | **69.5** | 807 |
| `no_cosine` | 1.2791 | 0.1649 | 19,141 | 1923 |
| `molf_style` | 1.3657 | 0.3026 | 1,507,660 | 0 |
| `full_ft` | 1.4841 | 0.8725 | 2,642,412 | 0 |

TriHOPE is dramatically cheaper than `full_ft` (3852x fewer coords), `molf_style`
(2198x) and `lora_only` (88x) — but those are not the relevant comparisons, since
they also forget far more. Against the arms that match it:

- `surprise_gate_s4` opens **10x fewer** coordinates (69.5M vs 686M), forgets
  **less** (0.0743 vs 0.0805), and gives up only 0.0113 of acquisition.
- `frozen_blocks` opens **zero** routed coordinates for acquisition within 0.0087.

So the 686M coordinates our controller commits do not buy efficiency either.
**Wall-clock is unusable here**: 63.0 min for `trihope` vs 88–89 min for the
others reflects how many runs shared the GPU during each batch, not the method.

## S2 — functional interference: REFUTES the value of routing

Pre-registered rule: "*Refutes the value of routing if `frozen_blocks` forgetting
matches or beats `trihope`.*" It matches — 0.0841 ± 0.0152 vs 0.0805 ± 0.0101.

`frozen_blocks` has **0 of 440,401,920 routed coordinates changed** and still
reproduces our full forgetting number. So forgetting in this setting travels
through the shared parameters, not through the routed weights our tri-store
protects. Theorem 1 guarantees those weights are immutable; that immutability is
not functional invariance, and the channel it closes is not the channel that
carries the damage.

## The one genuinely positive mechanism finding

`no_cosine` opens **19,141M coordinates against `trihope`'s 686M — 28x more** —
and its forgetting doubles (0.1649 vs 0.0805). Our cosine term is therefore doing
volume control, not allocation: it decides *how much* gets written, and write
volume is what tracks forgetting (+0.74 across the grid). That is consistent with
Codex's "plasticity throttle" hypothesis and inconsistent with the paper's
selection story. It is a real, mechanistic, defensible claim — and it is a claim
about a throttle, not about intelligent routing.

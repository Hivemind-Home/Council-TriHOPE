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

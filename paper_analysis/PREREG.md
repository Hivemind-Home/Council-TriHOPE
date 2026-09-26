# Pre-declared analysis protocol (committed 2026-09-23, before any final-batch run landed)

This file fixes, in advance, how the ICLR-2027 paper reads its experiments. Its git commit time is
the pre-declaration: the final GPU batch (`trihope` branch `campaign/iclr-2027-final-batch`)
had not produced a single result when it was committed. Anything decided after that point is
reported as post hoc.

## Data
- **Source.** Live-KD runs only: `results_live/` of `trihope`, plus the final batch. Every
  arm distils from four live teachers (τ = 4 KL, λ_KD = 1) plus 0.5 × CE on the cached teacher text.
- **Cache-mode campaign.** The cache-mode campaign (KD inert) appears only in an appendix, labelled
  as confounded. Relative to live mode it also differs in λ_CE, the routing thresholds, and the C̄
  gates.
- **Excluded arms, with reasons:**
  - `gold_ce`: evaluated on different (gold) targets.
  - live `gradient_routing`: crashed at step 3649 and was resumed with a changed override.
  - `plateau_trigger`: never fired, so it is identical to `lora_only`. It is reported as that
    fact, not as a baseline.
- **Pooling.** Runs are pooled only when their resolved configs are identical except for seed and
  paths (`scripts/check_config_drift.py`).

## Metrics
- **Primary, M1 (forgetting).** For each domain d ∈ {general, code, math, medical}, take the largest
  retention delta Δ_d over the phase boundaries after d was last trained. Δ_d is eval loss now
  minus eval loss at the end of d's last training phase. M1 is the mean of these four values.
  This is the computation in `analysis/tables.py`. Medical contributes one delta and math two.
- **Co-primary, plasticity.** Final macro validation loss: the mean over the four domains at step 5999.
- **Cost.** Routed permanent writes, in coordinate·steps on routed base weights: commit coordinates
  plus the base size of each merged block. Reported both nominally and effectively; a merge is
  effective only if the adapter received a write since its last reset. Shared parameters (tied
  embedding/LM head, norms) are excluded from this count and flagged in their own column.
- **Secondary metrics:**
  - M1 taken as the maximum over domains instead of the mean.
  - Mean retention delta.
  - Own-phase loss per domain.
  - Anytime macro loss (the mean over all eval points).
  - Steps to recover on `code_revisit`.
  - Exact match, in the appendix only (64 probes).

## Reference pool and noise
- **Reference pool.** Every run whose resolved config is identical to live E1 `trihope`. That is 7
  runs now (E1 ×3, `r_tier/trihope_replay` ×3, `ablation_grid/trihope` ×1), plus the
  `trihope_sentinel` runs if they pass the drift check.
- **What seeds cover.** Seeds vary initialisation and GPU nondeterminism only; the data order is
  fixed (`data.seed` = 42). The paper states this.
- **Resolution thresholds.** δ_M1 = 0.02 (already recorded in `CAMPAIGN_NOTES` §6, 2026-09-14).
  δ_L = 0.015 for final loss (2 × the mean gap between identical reruns). A point with n = 1
  needs 2δ.

## Tests
- **Resolved.** A difference is called resolved when |Δ| ≥ δ **and** an exact one-sided
  permutation test against the pool gives p ≤ 0.05. Anything else is called "level with".
  Pairwise wins over seeds (k/9) are reported alongside.
- **Primary family.** One-sided tests of "forgets more than the trihope pool", Holm-corrected
  across the four:
  - H1 `random_routing`
  - H2 `p_only` (always commit)
  - H3 `no_cosine`
  - H4 `molf_style_a0p5` (MoLF-SNR at 0.5)
- **Non-inferiority.** Tested for `frozen_blocks`, `no_consolidation` and `surprise_gate`. The upper
  end of the 90% Welch CI of (pool − control) on M1 must lie below δ_M1.
- **Everything else.** Effect sizes with 95% Welch CIs, and no p-values in the main text.
- **Crashed runs.** A crashed run is restarted from scratch with identical overrides and is never
  resumed with changed ones.
- **Diverged runs.** A diverged run is reported, not dropped, and medians are shown alongside means.

## Framing decision rule (DP1; fixed now, applied when `frozen_blocks` ×3 lands)
The wording of the claim about gated permanence follows from how `frozen_blocks` compares with the
trihope pool on M1 and final loss:

| Case | Outcome | Paper wording |
|---|---|---|
| (a) | `frozen_blocks` forgets more (Δ ≥ δ) **and** learns worse (Δ ≥ δ_L) | "gated routed writes add plasticity and reduce forgetting" |
| (b) | M1 level, but final loss worse by ≥ δ_L | "retention-neutral plasticity" |
| (c) | `frozen_blocks` forgets clearly less | exchange-rate framing: the best retention per unit of plasticity among rules that write permanently; drop "no cost" |
| (d) | both metrics level | lead with damage avoidance; add the blocks-only arm in Tier C |

## Freeze
- **Main-text data freeze:** Thu 2026-09-24, 12:00 UTC.
- **Appendix freeze:** Fri 2026-09-25, 12:00 UTC.

## Addendum 1 (committed 2026-09-23, after an internal review and before any result of the new arm)
- **Why.** An internal review pointed out that `random_routing` does more than randomise commits. It
  relabels *all* decisions, so it also defers rows from recurring buckets and replays about 4.7×
  more rows than the method. Each replay step also updates the adapters and the shared parameters.
  H1 therefore tests the joint choice of what to commit and what to replay. It does not test commit
  selection alone.
- **New arm.** `random_commit` (trihope `priority_s4`, 3 seeds). The controller classifies
  exactly as the method does, so every \defer (and every replay it triggers) is the method's own.
  The method's commits are then withheld, and random non-deferred selected blocks are committed at
  the method's per-phase commit share.
- **H1b.** One-sided test that `random_commit` forgets more than the trihope pool (M1), using the
  same exact-permutation test. It is added to the confirmatory family, and Holm now corrects over
  five tests (H1, H1b, H2, H3, H4).
- **Reporting.** Paper claims that "which updates become permanent matters" rest on H1b. H1 is
  reported as a routing-level control, and its replay-volume confound is stated. Replayed rows per
  arm are reported alongside every comparison.
- **Correction.** A single-seed arm compared against a pool of 7 has a minimum attainable exact
  p of 1/8. Under this protocol, single-seed arms can therefore never be "resolved". They are
  reported as exploratory.

## Addendum 2 (committed 2026-09-23): correction of a factual error above
- **Error.** The section "Reference pool and noise" says that seeds vary only initialisation and
  GPU nondeterminism and that the data order is fixed. That is wrong.
- **What the code actually does.** `stream.seed` is null, so the stream falls back to `train.seed`.
  The seed therefore chooses which buckets recur in each phase, which rows fill each step, and the
  novel rows. Stream digests differ across seeds 1337, 2024 and 7.
- **What is fixed.** The phase structure and the validation rows (`data.seed = 42`).
- **What same-seed reruns measure.** Same-seed reruns see an identical stream, so their spread is
  GPU nondeterminism alone.
- **Effect on the protocol.** No test or threshold changes. Only the description of what the seeds
  vary is corrected, in this file and in the paper.

## Addendum 3 (committed 2026-09-24, after the final batch): outcomes under the rules above
Written after the final-batch data arrived. It applies the pre-declared rules; it changes no test,
threshold or family.
- **What ran.** Every arm marked "never cut" finished (16 runs, 0 failed): `frozen_blocks` ×3,
  `trihope_sentinel` ×3, `random_commit` ×3, `surprise_gate_s4` ×2, `molf_style_a0p7` ×2,
  `no_cosine` ×2, `pgate_c0p35` ×1. Cut for compute before running: extra seeds of `molf_style_a0p5`,
  `moments_optimizer`, `no_surprise`, `topm_2`, `topk_25`, `topm_all`, and `pgate_c0p2/c0p65/c0p8`.
  `periodic_merge` was stopped part-way; its runs have no summary and are not reported.
- **Config drift.** `check_config_drift.py` is clean for all 16 runs. Each new seed matches its
  seed-1337 counterpart exactly, so the arms pool to n = 3.
- **DP2 (sentinel).** All three sentinels fall inside the pool's rerun spread; they join the pool,
  which is now n = 10 (M1 0.0838 ± 0.0060, final loss 1.2461 ± 0.0043).
- **Confirmatory family (Holm over five).** H1 random relabel p = 0.017; **H1b random commit
  p = 0.472 (not significant)**; H2 always-commit p = 0.017; H3 no C̄ gate p = 0.017 (now n = 3);
  H4 SNR at 0.5 p = 0.182 (still n = 1).
- **DP4 / H1b.** Not significant. As the reporting rule above requires, the paper makes **no**
  claim that *which* updates become permanent matters. It reports that random commits at the
  method's rate and replay volume forget the same. The 90% Welch interval for that difference
  ([-0.007, +0.006]) is reported descriptively; `random_commit` was not in the pre-declared
  non-inferiority list. H1's excess forgetting is attributed to its replay volume (4.8× the
  method's), the confound anticipated in Addendum 1.
- **DP1 (`frozen_blocks`).** M1 level (difference +0.000; non-inferiority 90% upper bound 0.019,
  below δ_M = 0.02). Final loss 0.012 worse than the pool (p = 0.007), which is below δ_L = 0.015
  and therefore level. Both axes level: case (d). Wording: the method's routed writes add no
  measurable forgetting beyond training the shared parameters; the plasticity difference is
  reported with its p-value but not claimed. Theorem 1 at scale: 0 of 440,401,920 routed
  coordinates changed in all three runs (T3).
- **DP3.** `surprise_gate_s4` (n = 3): M1 −0.010 (p = 0.010, below δ_M, level); final loss +0.016,
  resolved worse. `molf_style_a0p7` (n = 3): M1 level; final loss +0.045, resolved worse.
- **Also reported, level under the rule.** `no_consolidation` forgets more than the pool in 30/30
  pairings (+0.019, p = 0.003), just below δ_M; the paper states it as level and suggestive.

## Addendum 4 (committed 2026-09-25): results that arrived after Addendum 3
Written after the data; applies the rules above and changes no test, threshold or family.
- **`periodic_merge` ×3 completed.** The runs stopped part-way on 2026-09-24 were rerun from
  scratch (single launch, 2026-09-25 06:39 UTC; no resume). Config drift is clean. Forgetting 2.5×
  the pool (0.207, resolved worse); final loss 1.215 (resolved better). Reported in the main table.
- **T3 weight drift on every new run.** Shared-only: 0 of 440,401,920 routed coordinates changed in
  all three runs. Surprise-only at θ_S = 4, which never opens a routed block but replays into
  adapters: one of two checked runs is bit-identical; the other has 2 routed coordinates changed
  (the larger by 0.00195, one bf16 unit) with no logged write. The optimizer's closed-coordinate path
  and the merge path are both excluded; the cause is unknown. Disclosed in the paper as an open
  discrepancy. (Several `weight_drift.json` files carry a stale tail after the first JSON object;
  the pipeline reads the first object.)
- **Acquisition (pre-declared secondary metric "own-phase loss").** Reported for every arm as the
  mean over code, math and medical of each domain's loss at the end of its own phase, from the
  per-phase evaluation history. No δ was declared for it, so it is reported with pairwise wins and
  exact permutation p, never as "resolved". Note: the `run_summary.json` field
  `retention.own_phase_loss` is *not* this quantity; the mixed tail retrains every domain and
  overwrites it, so it equals the final per-domain loss. Main observations: shared-only has higher
  own-phase loss than the pool in 30/30 pairings (1.136 vs 1.111); random commit ties (1.112);
  never-commit ties (1.114) while forgetting more in 30/30 pairings; surprise-only θ_S = 4 is higher
  in 26/30; the SNR rule at 0.7 in 28/30; no-defer-tier in 60/60.
- **Still missing.** Extra seeds of `molf_style_a0p5` and `moments_optimizer` were launched but had
  not produced metrics when this addendum was written; H4 stays at n = 1.
- **Outside this protocol.** Experiments designed on 2026-09-25 after these results were seen
  (`quota_shuffle` / `yoked_2x2`, `order2`, `student17`) are not part of the pre-declared family. If
  reported, they are exploratory.

## Addendum 5 (committed 2026-09-25, evening): exploratory runs, reported outside the protocol
- **Where they come from.** The operator's session designed `quota_shuffle` / `yoked_2x2`,
  `order2` and `student17` on 2026-09-25, after the results above were known. Runs finished on the
  GPU box (`yoked_blind_lo`, `yoked_blind_hi`, seed 1337 only) and on rented A100s via Modal
  (`yoked_full_lo` seeds 1337/2024, `trihope17` seed 1337) before the campaign was stopped at
  15:10 UTC; `yoked_full_hi` ×3 was killed at ~5300/6000 steps and is unusable; `order2` never ran.
- **How they are used.** Excluded from every table and test. Reported once, in an appendix
  paragraph, with the pre-registered M1 and own-phase definitions: 1.7B TriHOPE M1 0.086, own-phase
  0.934 (no controls at 1.7B, so no claim about the controller there); the two A100 reruns of the
  TriHOPE configuration M1 0.089 and 0.098 (pool range 0.075–0.092); the shuffled cells M1 0.121 and
  0.090 at n = 1 with 1.9× and 2.2× TriHOPE's commits, so the shuffle does not hold volume fixed.
- **Metric caution.** The operator's `analyze_yoked.py` and `results_modal/README.md` use "worst
  rise above the best loss so far" as forgetting and `retention.own_phase_loss` (= final loss) as
  acquisition. Neither is the pre-registered definition; the paper does not use those numbers.

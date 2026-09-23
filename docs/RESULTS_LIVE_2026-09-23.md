# LIVE campaign results — KD active

Generated 2026-09-23. 115 live runs, 0 failed.

**Why this supersedes the cache-mode tables.** In cache mode `lambda_kd` is inert without
logit caches (`loss/kd = 0.0`, `kd_row_hit_frac = 0.0`), so every cached number measured
CE + regularisation only. Live is the only configuration where the distillation term the
paper is about is non-zero (`loss/kd = 4.199`). Thresholds also differ (docs §10).

Metric: mean over {general, code, math, medical} of `worst_retention_delta`. Lower is better.

## E1 baselines (n=3)

| arm | live | cache | change |
|---|---:|---:|---|
| `gold_ce` | 0.0596 ± 0.009 | 0.0529 | worse |
| `trihope` | 0.0805 ± 0.010 | 0.1132 | **better** |
| `surprise_gate` | 0.0897 ± 0.016 | 0.1436 | **better** |
| `trihope_no_hash` | 0.0936 ± 0.017 | 0.1331 | **better** |
| `no_retrieval` | 0.0959 ± 0.013 | 0.0706 | worse |
| `no_consolidation` | 0.1031 ± 0.017 | 0.1104 | **better** |
| `random_routing` | 0.1645 ± 0.073 | 0.3011 | **better** |
| `plateau_trigger` | 0.1825 ± 0.013 | 0.1823 | worse |
| `lora_only` | 0.1828 ± 0.013 | 0.1841 | **better** |
| `molf_style` | 0.3026 ± 0.076 | 0.4587 | **better** |
| `full_ft` | 0.8725 ± 1.019 | 3.0951 | **better** |

## E5 bad-teacher (n=3)

| arm | live | cache | change |
|---|---:|---:|---|
| `trihope_lowconf` | 0.0769 ± 0.010 | 0.1100 | **better** |
| `trihope_nogate` | 0.0795 ± 0.013 | 0.1086 | **better** |
| `trihope` | 0.0840 ± 0.011 | 0.1086 | **better** |
| `trihope_forced` | 0.0863 ± 0.016 | — | — |
| `gradient_routing` | 0.1264 ± 0.007 | 1.2211 | **better** |
| `molf_snr` | 0.1295 ± 0.026 | 0.0407 | worse |
| `lora_only` | 0.1764 ± 0.015 | 0.1736 | worse |
| `molf_style` | 0.2514 ± 0.044 | 0.6876 | **better** |
| `full_ft` | 0.2703 ± 0.044 | 2.5914 | **better** |

## E4 ablation grid (n=1)

| arm | live | cache | change |
|---|---:|---:|---|
| `topm_2` | 0.0742 | 0.0692 | worse |
| `topk_25` | 0.0754 | 0.0755 | **better** |
| `no_volatility` | 0.0805 | 0.0975 | **better** |
| `consolidation_strict` | 0.0876 | 0.0909 | **better** |
| `trihope` | 0.0881 | 0.0961 | **better** |
| `no_repetition` | 0.0887 | 0.0514 | worse |
| `topk_100` | 0.0934 | 0.1033 | **better** |
| `no_teacher_conf` | 0.0948 | 0.1063 | **better** |
| `stability_instant` | 0.0965 | 0.1195 | **better** |
| `no_surprise` | 0.1147 | 0.1897 | **better** |
| `no_cosine` | 0.1506 | 0.2078 | **better** |
| `topm_all` | 0.2036 | 0.1392 | worse |
| `moments_optimizer` | 0.2234 | 0.2498 | **better** |

## What changed, and it is the headline

**The fair MoLF baseline no longer beats the method.** E5, n=3:

- live: `trihope` **0.0840 ± 0.011** vs `molf_snr` 0.1295 ± 0.026
- cache: `trihope` 0.1086 vs `molf_snr` 0.0407

In cache mode molf_snr was 2.8x better; with KD active TriHOPE is ~1.5x better, error bars
separated. The earlier negative result was an artefact of distillation being switched off.

**E1 reverses too.** `no_retrieval` beat trihope in cache (0.0706 vs 0.1132); live it loses
(0.0959 vs 0.0805). `no_consolidation` likewise.

**What survives unchanged.** The controller signals remain load-bearing: `no_cosine` +0.0625, `no_surprise` +0.0265, `moments_optimizer` +0.1353 against the in-grid baseline.

**What still stands against the method.** Both write-budget reductions still beat the shipped
config (`topm_2` 0.0742, `topk_25` 0.0754 vs 0.0881) — it still over-writes. `trihope_nogate` and `trihope_lowconf` still sit level with or above `trihope` in E5, so the confidence gate remains unproven.

## Limits

- E4 is n=1 per arm; differences under ~0.006 are noise (see the cache-mode noise floor).
- The live rollback/full-restore sub-study was dropped: `checkpoint_before_merge` with 153 live
  consolidation events is ~600 GB per run. Cache mode retains that sub-study.
- Live and cache differ in thresholds as well as KD, so the two tables are not paired.
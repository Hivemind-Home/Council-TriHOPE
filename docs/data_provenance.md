# Data provenance and supervision mode — audit of 2026-09-05

What the four Layer-C datasets actually contain, checked against the hub
cards and the parquet rows themselves, and what that means for how the
paper describes its supervision. Companion to `docs/dataset_preperation.md`
(the schema design) and `docs/experiments.md` (how the data is used).

## 1. Who wrote the teacher text

| Domain | `teacher_id` in the rows | Who ran that model | Evidence |
|---|---|---|---|
| general | `general_teacher_deepseek_r1_distill_llama_70b` | Glaive, upstream | Card: "traces shipped with the source corpus" (`glaiveai/reasoning-v1-20m`). Full traces with `<think>` tags, ~6k characters. |
| code | `code_teacher_deepseek_r1` | NVIDIA, upstream | Same wording; source `nvidia/OpenCodeReasoning`. Full traces, up to ~33k characters. |
| math | `math_teacher_deepseek_r1_distill_qwen_1p5b` | this project | No card text. 90 % of outputs open in the R1-distill register ("Okay, so …"); confidence tracks −entropy with correlation 0.99, i.e. computed from token logits. |
| medical | `medical_teacher_qwen2p5_1p5b_instruct_clean` | this project | No card text. Instruct-style answers; 6 % of rows at confidence 0.0; mean confidence 0.59. |

The general and code cards describe a DataFlow-style curation chain
"with DeepSeek as the LLM serving backend" whose step 4 is an **LLM audit
gate**: `deepseek-v4-flash` grades a random sample per batch for prompt
clarity, answer relevance, reasoning soundness and truncation. That is
quality control on a sample, not generation; the teacher traces are marked
"otherwise verbatim".

**Not verifiable from here.** The generation and curation scripts are not
in this repository, the math and medical cards carry no description, and
the per-row `teacher_logits_path` files (`teacher_logits/<domain>/<split>/
<uuid>_<domain>.npz`) are not published anywhere. The math/medical
attribution rests on the ids, the config comment in `stream_headline.yaml`
("exactly the models that produced the cached supervision"), and the
statistics above. **Before camera-ready: obtain the generation logs from
whoever built the datasets and replace this paragraph with the exact
models per domain.** If a hosted model wrote any of the math or medical
rows, the paper must say so; a reviewer who samples the public dataset can
see the register.

Licences: the two upstream corpora are Apache-2.0 (general) and CC-BY-4.0
(code). Outputs of some hosted APIs carry restrictions on training other
models — settle this once the generation logs are in hand.

## 2. What "teacher confidence" is

- **math / medical:** derived from the teacher's own token logits (see the
  entropy correlation above). A per-answer summary of a real distribution.
- **general / code:** the upstream traces came without logits, so the
  confidence column was produced by an undocumented scoring pass. It is a
  *dataset-provided score*, not the authoring model's entropy.

The confidence gate (`use_teacher_confidence`, `ce_confidence_weighting`)
and the `no_teacher_conf` ablation should be read with that in mind. The
medical phase (mean 0.59, 6.5 % of rows below 0.5) is the only phase where
the gate fires materially; it is also the one domain where the score is
logit-derived.

## 3. What the student is trained on

**Cache mode (E1–E5, every figure in the paper).** The KD term is inert:
no logit files exist, so the collator emits an all-zero `teacher_logits_
mask` and `loss/kd` is exactly 0 for the whole run. The objective is
confidence-weighted cross-entropy on `teacher_output_text` — **sequence-
level distillation** (Kim & Rush, 2016), the same recipe that produced the
R1-distill teachers themselves. Every controller sees the identical
gradient stream, so E1–E5 comparisons are unaffected by the loss type.

**Live mode (Tier 4, `configs/experiments/live_kd_small.yaml`).** Four
frozen Qwen-vocabulary teachers score the *same* cached text under teacher
forcing and the student matches their full distributions (τ² · KL at
τ = 4) plus CE at half weight — the Hinton-style objective the theory
document specifies. Two caveats: (i) the vocabulary constraint means the
general and code live teachers (Qwen3-4B, Qwen2.5-Coder-1.5B) are *not* the
models that authored those traces, so they score another model's answer;
(ii) it is therefore a different stream and belongs in the appendix as a
robustness check, not in Figure 1.

**Truncation.** The math teacher outputs are cut at roughly 1000
characters (only 5.6 % contain a closing `</think>`); the general and code
traces are complete but the student's `max_seq_len` is 512 tokens (1024
in the headline profile), so it sees the *opening* of a reasoning trace,
not full solutions. One sentence in limitations.

## 4. Wording the paper must use

- "Sequence-level distillation from cached teacher traces"; never
  "temperature-scaled KL" for the main results (the KL is only active in
  the Tier 4 appendix run).
- "Teacher confidence is a dataset-provided per-answer score whose
  derivation differs by domain" — not "teacher entropy".
- Setup table listing, per domain, the source corpus, the authoring model
  as far as it is documented, and whether logits were available.
- Teacher *selection* is not a contribution; in cache mode the router
  reads the row's `teacher_id` (the cosine-prototype router of Theory 201
  §1 runs only on synthetic data). Teacher *accountability* (E5) is.

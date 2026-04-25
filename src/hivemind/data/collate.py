"""Distillation collator: raw dataset rows → training tensors.

Each sample becomes ``[prompt_tokens] [response_tokens]`` with ``labels``
masked to ``-100`` on the prompt and pad. A single response per row keeps
the forward pass single-pass and keeps CE / KD aligned on the same span.

Per-row response selection:

* If ``teacher_output_text`` is present → response = teacher text. This is
  the canonical "distillation" setup and is what the NPZ logits
  (``teacher_logits_path``) were computed over, so KD aligns when the
  cache hits.
* Else if ``target_text`` is present → response = gold text. KD is
  disabled for this row via ``teacher_logits_mask[b] = 0``.
* Otherwise the row is dropped from the batch.

Emitted batch keys (see also ``training.py::_unpack_batch``):

* ``input_ids`` ``[B, T]``
* ``attention_mask`` ``[B, T]``
* ``labels`` ``[B, T]`` — ``-100`` on prompt / pad
* ``teacher_logits`` ``[B, T, V]`` — zeros when unavailable
* ``teacher_logits_mask`` ``[B]`` float — 0 or 1 per row
* ``sample_ids`` list[str]
* ``metadata`` dict of list[str] — ``domain``, ``bucket_id``,
  ``teacher_id``, ``label_source`` (``"teacher"`` / ``"gold"``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .teacher_cache import TeacherLogitsCache
from .tokenizer import TokenizerProtocol


@dataclass
class CollateOutput:
    """Typed mirror of the collator's return dict — for documentation only."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    teacher_logits: torch.Tensor
    teacher_logits_mask: torch.Tensor
    sample_ids: list[str]
    metadata: dict[str, list[Any]]


IGNORE_INDEX = -100


class DistillCollator:
    """Tokenize + pad a batch of HivemindHFDataset rows."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        max_seq_len: int,
        logits_cache: TeacherLogitsCache,
        prompt_max_fraction: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.logits_cache = logits_cache
        self.prompt_max_fraction = prompt_max_fraction
        self.pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    # -- row-level ---------------------------------------------------------

    def _encode(self, text: str, max_len: int) -> list[int]:
        if max_len <= 0:
            return []
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[:max_len]

    def _pick_response(self, row: dict[str, Any]) -> tuple[str | None, str]:
        teacher_text = row.get("teacher_output_text")
        if teacher_text:
            return teacher_text, "teacher"
        target_text = row.get("target_text")
        if target_text:
            return target_text, "gold"
        return None, "none"

    def _build_row(
        self, row: dict[str, Any]
    ) -> dict[str, Any] | None:
        input_text = row.get("input_text")
        if not input_text:
            return None

        response, label_source = self._pick_response(row)
        if response is None:
            return None

        prompt_budget = max(1, int(self.max_seq_len * self.prompt_max_fraction))
        prompt_ids = self._encode(input_text, prompt_budget)
        if not prompt_ids:
            return None

        response_budget = self.max_seq_len - len(prompt_ids)
        if response_budget <= 0:
            return None
        response_ids = self._encode(response, response_budget)
        if not response_ids:
            return None

        seq = prompt_ids + response_ids
        labels_row = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

        # KD only aligns when we actually used the teacher's text
        teacher_tensor: torch.Tensor | None = None
        if label_source == "teacher":
            teacher_tensor = self.logits_cache.get(row.get("teacher_logits_path"))

        try:
            confidence = float(row.get("teacher_confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        try:
            entropy = float(row.get("teacher_entropy", 0.0))
        except (TypeError, ValueError):
            entropy = 0.0

        return {
            "input_ids": seq,
            "labels": labels_row,
            "response_span": (len(prompt_ids), len(seq)),
            "teacher_tensor": teacher_tensor,
            "sample_id": row.get("sample_id", ""),
            "label_source": label_source,
            "domain": row.get("domain", ""),
            "bucket_id": row.get("bucket_id", ""),
            "teacher_id": row.get("teacher_id", ""),
            "teacher_confidence": confidence,
            "teacher_entropy": entropy,
        }

    # -- batch assembly ----------------------------------------------------

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        built: list[dict[str, Any]] = []
        for row in batch:
            b = self._build_row(row)
            if b is not None:
                built.append(b)

        if not built:
            # Preserve DataLoader invariants by raising — an empty batch
            # signals every row was malformed, which is upstream's bug.
            raise ValueError(
                "DistillCollator: all rows in the batch were missing input_text / response text."
            )

        B = len(built)
        T = min(self.max_seq_len, max(len(r["input_ids"]) for r in built))

        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)

        V = self.vocab_size
        teacher_logits = torch.zeros((B, T, V), dtype=torch.float32) if V > 0 else torch.zeros((B, T, 1))
        teacher_logits_mask = torch.zeros((B,), dtype=torch.float32)

        sample_ids: list[str] = []
        meta: dict[str, list[Any]] = {
            "domain": [],
            "bucket_id": [],
            "teacher_id": [],
            "label_source": [],
        }
        teacher_confidence = torch.zeros((B,), dtype=torch.float32)
        teacher_entropy = torch.zeros((B,), dtype=torch.float32)

        for b_idx, row in enumerate(built):
            seq = row["input_ids"][:T]
            lab = row["labels"][:T]
            L = len(seq)
            input_ids[b_idx, :L] = torch.tensor(seq, dtype=torch.long)
            attention_mask[b_idx, :L] = 1
            labels[b_idx, :L] = torch.tensor(lab, dtype=torch.long)

            tens = row["teacher_tensor"]
            if tens is not None and V > 0 and tens.shape[-1] == V:
                start, end = row["response_span"]
                end = min(end, T)
                span = max(0, end - start)
                # Clamp to the NPZ's token count — a shorter cache just
                # means fewer KD-active positions, not a shape error.
                span = min(span, int(tens.shape[0]))
                if span > 0:
                    # Teacher NPZ is over the response tokens only; align to the
                    # prompt offset so KD is computed on the same positions CE is.
                    teacher_logits[b_idx, start:start + span] = tens[:span]
                    teacher_logits_mask[b_idx] = 1.0

            sample_ids.append(row["sample_id"])
            meta["domain"].append(row["domain"])
            meta["bucket_id"].append(row["bucket_id"])
            meta["teacher_id"].append(row["teacher_id"])
            meta["label_source"].append(row["label_source"])
            teacher_confidence[b_idx] = row["teacher_confidence"]
            teacher_entropy[b_idx] = row["teacher_entropy"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "teacher_logits": teacher_logits,
            "teacher_logits_mask": teacher_logits_mask,
            "teacher_confidence": teacher_confidence,
            "teacher_entropy": teacher_entropy,
            "sample_ids": sample_ids,
            "metadata": meta,
        }

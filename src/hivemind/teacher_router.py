"""Per-sample teacher routing.

Two routers are supported:

* ``TeacherRouter`` (cosine) — Theory 201 §1: cosine similarity in the
  shared embedding space between a routing vector and per-teacher
  prototypes. Default for synthetic training and for ablations.
* ``MetadataRouter`` — reads ``teacher_id`` / ``domain`` from the batch
  metadata and maps it to a teacher index. Default for HF mode because
  the dataset already prescribes the right teacher per sample; running
  cosine on top re-introduces noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RouterConfig:
    """Teacher router configuration."""

    hard_routing: bool = True  # argmax (hard) vs soft mixture


class TeacherRouter:
    """Per-sample teacher selection via cosine similarity."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route samples to teachers.

        Args:
            embeddings: [B, d] sample routing embeddings (L2-normalized).
            prototypes: [K, d] teacher prototype vectors (L2-normalized).

        Returns:
            teacher_indices: [B] index of selected teacher per sample.
            scores: [B, K] cosine similarity scores.
        """
        # Cosine similarity (both inputs assumed L2-normalized)
        scores = embeddings @ prototypes.T  # [B, K]

        if self.config.hard_routing:
            teacher_indices = scores.argmax(dim=-1)  # [B]
        else:
            # Soft routing: sample from softmax distribution
            probs = F.softmax(scores, dim=-1)
            teacher_indices = torch.multinomial(probs, 1).squeeze(-1)

        return teacher_indices, scores


class MetadataRouter:
    """Route by dataset-provided ``teacher_id`` / ``domain``.

    Lookup order per sample:

    1. ``metadata['teacher_id'][i]`` matched against registered teacher
       names (which come from ``TeacherInfo.name``).
    2. If that misses, ``metadata['domain'][i]`` matched against the
       ``domain`` attribute on ``CacheBackedTeacher``.
    3. If both miss, fall back to index 0 (and record the miss count —
       the caller logs it). Falling back beats raising because a single
       unroutable sample shouldn't kill the whole step.
    """

    def __init__(
        self,
        teacher_names: list[str],
        domain_to_index: dict[str, int] | None = None,
    ) -> None:
        self._name_to_index = {name: i for i, name in enumerate(teacher_names)}
        self._domain_to_index = dict(domain_to_index or {})
        self.miss_count = 0

    def route(self, metadata: dict[str, list[Any]], device: torch.device) -> torch.Tensor:
        teacher_ids = metadata.get("teacher_id", [])
        domains = metadata.get("domain", [])
        B = len(teacher_ids) if teacher_ids else len(domains)
        indices = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(B):
            tid = teacher_ids[i] if i < len(teacher_ids) else ""
            dom = domains[i] if i < len(domains) else ""
            idx = self._name_to_index.get(tid)
            if idx is None:
                idx = self._domain_to_index.get(dom)
            if idx is None:
                idx = 0
                self.miss_count += 1
            indices[i] = idx
        return indices


def batch_teacher_forward(
    tokens: torch.Tensor,
    teacher_indices: torch.Tensor,
    teachers: list,
    vocab_size: int,
    cached_logits: torch.Tensor | None = None,
    cached_logits_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble per-sample teacher logits.

    When ``cached_logits`` is provided (HF / cache-backed path), this is a
    fast pass-through: cached rows use the precomputed tensor, and any
    row whose ``cached_logits_mask`` is 0 falls through to live inference
    on that row's assigned teacher. In the common HF case every row is
    cached so no live forward runs.

    When no cache is provided, grouping-by-teacher live inference is used,
    which is the synthetic-teacher path.

    Args:
        tokens: ``[B, T]`` input token indices.
        teacher_indices: ``[B]`` teacher index per sample.
        teachers: list of TeacherInfo objects.
        vocab_size: vocabulary size for output tensor.
        cached_logits: optional ``[B, T, V]`` precomputed teacher logits.
        cached_logits_mask: optional ``[B]`` float mask, 1 = use cache.

    Returns:
        ``[B, T, vocab_size]`` teacher logits.
    """
    B, T = tokens.shape
    device = tokens.device
    teacher_logits = torch.zeros(B, T, vocab_size, device=device)

    # Fast path: fully cached.
    if cached_logits is not None and cached_logits_mask is not None:
        if cached_logits.shape[-1] == vocab_size:
            cached = cached_logits.to(device=device, dtype=teacher_logits.dtype)
            mask = cached_logits_mask.to(device=device).view(B, 1, 1)
            teacher_logits = teacher_logits + cached * mask
        # Rows where cache is missing fall through to live forward below.
        live_mask = cached_logits_mask.to(device=device) < 0.5
    else:
        live_mask = torch.ones(B, dtype=torch.bool, device=device)

    if not live_mask.any():
        return teacher_logits

    with torch.no_grad():
        for k in range(len(teachers)):
            selector = (teacher_indices == k) & live_mask
            if not selector.any():
                continue
            subset = tokens[selector]
            subset_logits = teachers[k].model(subset)
            teacher_logits[selector] = subset_logits

    return teacher_logits

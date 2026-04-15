"""Per-sample teacher routing via cosine similarity.

Routes each training sample to the best teacher by comparing the sample's
routing embedding against teacher prototypes in the shared embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def batch_teacher_forward(
    tokens: torch.Tensor,
    teacher_indices: torch.Tensor,
    teachers: list,
    vocab_size: int,
) -> torch.Tensor:
    """Efficiently batch teacher forward passes by grouping samples per teacher.

    Instead of running each teacher on the full batch, groups samples by
    their assigned teacher and runs each teacher only on its subset.

    Args:
        tokens: [B, T] input token indices.
        teacher_indices: [B] teacher index per sample.
        teachers: list of TeacherInfo objects.
        vocab_size: vocabulary size for output tensor.

    Returns:
        teacher_logits: [B, T, vocab_size] teacher logits for each sample.
    """
    B, T = tokens.shape
    device = tokens.device
    teacher_logits = torch.zeros(B, T, vocab_size, device=device)

    with torch.no_grad():
        for k in range(len(teachers)):
            mask = teacher_indices == k  # [B]
            if not mask.any():
                continue
            subset_tokens = tokens[mask]  # [n_k, T]
            subset_logits = teachers[k].model(subset_tokens)  # [n_k, T, V]
            teacher_logits[mask] = subset_logits

    return teacher_logits

"""R-store: Retrieval memory for novel/one-off patterns.

Stores sample embeddings, teacher outputs, and metadata in a fixed-size
FIFO buffer. Supports cosine similarity search for the retrieval hit
rate repetition signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RetrievalEntry:
    """A single entry in the retrieval store."""

    embedding: torch.Tensor  # [d]
    teacher_id: int
    bucket_id: int
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStore:
    """Fixed-size FIFO retrieval memory buffer.

    Stores embeddings and metadata for samples routed to R-store.
    Supports nearest-neighbor queries for the retrieval hit rate signal.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buffer: deque[RetrievalEntry] = deque(maxlen=max_size)

    def add(self, entry: RetrievalEntry) -> None:
        """Add an entry to the retrieval buffer."""
        self._buffer.append(entry)

    def hit_count(self, embedding: torch.Tensor, threshold: float = 0.85) -> int:
        """Count entries with cosine similarity above threshold.

        Args:
            embedding: [d] query embedding.
            threshold: cosine similarity threshold.

        Returns:
            Number of matching entries.
        """
        if not self._buffer:
            return 0

        query = embedding.detach().float().flatten()
        embeddings = torch.stack([e.embedding.float().flatten() for e in self._buffer])
        sims = F.cosine_similarity(query.unsqueeze(0), embeddings)
        return (sims >= threshold).sum().item()

    def query_nearest(
        self, embedding: torch.Tensor, k: int = 5
    ) -> list[tuple[float, RetrievalEntry]]:
        """Find k nearest entries by cosine similarity.

        Returns:
            List of (similarity, entry) tuples, sorted by descending similarity.
        """
        if not self._buffer:
            return []

        query = embedding.detach().float().flatten()
        embeddings = torch.stack([e.embedding.float().flatten() for e in self._buffer])
        sims = F.cosine_similarity(query.unsqueeze(0), embeddings)

        k = min(k, len(self._buffer))
        top_sims, top_indices = sims.topk(k)

        return [
            (top_sims[i].item(), list(self._buffer)[top_indices[i].item()])
            for i in range(k)
        ]

    @property
    def size(self) -> int:
        return len(self._buffer)

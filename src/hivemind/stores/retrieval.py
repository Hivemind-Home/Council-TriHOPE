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
    """A single entry in the retrieval store.

    Theory 101 §7 prescribes storing the embedding, bucket id, teacher
    id, teacher soft targets, and metadata. We carry the embedding +
    bookkeeping fields directly; ``teacher_output_text`` holds the
    teacher's text response (the cheap, always-available proxy for
    "soft targets"); per-token logits would be too large to keep in
    memory and live in the on-disk teacher cache instead.
    """

    embedding: torch.Tensor  # [d]
    teacher_id: int
    bucket_id: int
    step: int
    teacher_name: str = ""
    teacher_output_text: str = ""
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

    def _aligned(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Query and stacked buffer embeddings, on one device.

        ``WriteExecutor`` stores entries with ``.cpu()`` so a full buffer
        does not pin GPU memory for ``max_size`` embeddings, but the query
        arrives on the training device. Without this the cosine similarity
        raises a device mismatch the moment the R-store is non-empty on a
        GPU run — invisible to the CPU test suite, which has both sides on
        the same device.

        The query moves to the buffer, never the reverse:
        ``_retrieval_hit_rate`` calls this once per probe row, so hauling
        ``max_size`` embeddings to the GPU each time would cost far more
        than moving one row off it.
        """
        embeddings = torch.stack([e.embedding.float().flatten() for e in self._buffer])
        query = embedding.detach().float().flatten().to(embeddings.device)
        return query, embeddings

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

        query, embeddings = self._aligned(embedding)
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

        query, embeddings = self._aligned(embedding)
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

    def state_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "entries": [
                {
                    "embedding": e.embedding.detach().cpu(),
                    "teacher_id": int(e.teacher_id),
                    "bucket_id": int(e.bucket_id),
                    "step": int(e.step),
                    "teacher_name": str(e.teacher_name),
                    "teacher_output_text": str(e.teacher_output_text),
                    "metadata": dict(e.metadata),
                }
                for e in self._buffer
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._buffer = deque(maxlen=self.max_size)
        for raw in state.get("entries", []):
            self._buffer.append(
                RetrievalEntry(
                    embedding=raw["embedding"],
                    teacher_id=int(raw["teacher_id"]),
                    bucket_id=int(raw["bucket_id"]),
                    step=int(raw["step"]),
                    teacher_name=str(raw.get("teacher_name", "")),
                    teacher_output_text=str(raw.get("teacher_output_text", "")),
                    metadata=dict(raw.get("metadata", {})),
                )
            )

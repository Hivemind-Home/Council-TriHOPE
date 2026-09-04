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


@dataclass(eq=False)
class RetrievalEntry:
    """A single entry in the retrieval store.

    Theory 101 §7 prescribes storing the embedding, bucket id, teacher
    id, teacher soft targets, and metadata. We carry the embedding +
    bookkeeping fields directly; ``teacher_output_text`` holds the
    teacher's text response (the cheap, always-available proxy for
    "soft targets"); per-token logits would be too large to keep in
    memory and live in the on-disk teacher cache instead.

    The optional ``input_ids`` / ``labels`` / ``kd_row_mask`` /
    ``teacher_confidence`` fields carry the representative row of the
    R-routed batch so it can be *replayed* into the fast store later
    (``RetrievalConfig.replay_on_hit``). They are ``None`` unless replay is
    enabled, which keeps the store's memory footprint unchanged by default.

    ``eq=False``: entries compare by identity. The generated field-wise
    ``__eq__`` would call ``tensor == tensor`` and raise on multi-element
    tensors the first time an entry is looked up or removed.
    """

    embedding: torch.Tensor  # [d]
    teacher_id: int
    bucket_id: int
    step: int
    teacher_name: str = ""
    teacher_output_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    input_ids: torch.Tensor | None = None  # [T] CPU, representative row
    labels: torch.Tensor | None = None  # [T] CPU, -100 on prompt / pad
    # KD row mask to use when replaying: ``None`` on the synthetic path (KD
    # against the live teacher), ``0.0`` for HF rows (KD is inert without a
    # logits cache, and cached logits are not stored — replay is CE-only).
    kd_row_mask: float | None = None
    teacher_confidence: float | None = None


class RetrievalStore:
    """Fixed-size FIFO retrieval memory buffer.

    Stores embeddings and metadata for samples routed to R-store.
    Supports nearest-neighbor queries for the retrieval hit rate signal.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buffer: deque[RetrievalEntry] = deque(maxlen=max_size)
        # Entries handed back to the trainer by ``pop_for_bucket`` over the
        # whole run (checkpointed, so ``run_summary`` is exact after resume).
        self.replayed_total: int = 0

    def add(self, entry: RetrievalEntry) -> None:
        """Add an entry to the retrieval buffer."""
        self._buffer.append(entry)

    # -- replay support (RetrievalConfig.replay_on_hit) ---------------------
    #
    # Linear scans over at most ``max_size`` (default 1000) Python objects:
    # tens of microseconds per step. A side index keyed by bucket would go
    # stale the moment ``deque(maxlen)`` silently evicts its oldest entry,
    # so none is kept.

    def entries_for_bucket(self, bucket_id: int) -> list[RetrievalEntry]:
        """All entries for ``bucket_id`` in insertion (FIFO) order."""
        b = int(bucket_id)
        return [e for e in self._buffer if e.bucket_id == b]

    def distinct_steps_for_bucket(self, bucket_id: int) -> list[int]:
        """Sorted distinct origin steps that wrote ``bucket_id``.

        The writer appends one entry per R *action*, i.e. up to Top-M
        identical entries per step, so recurrence is counted in steps.
        """
        return sorted({int(e.step) for e in self.entries_for_bucket(bucket_id)})

    def pop_for_bucket(self, bucket_id: int, n: int) -> list[RetrievalEntry]:
        """Remove and return the ``n`` oldest steps' entries for a bucket.

        Returns one representative entry (the first appended) per origin
        step, oldest step first, and removes *every* entry of those steps
        for that bucket so a replayed row is never replayed again.
        """
        b = int(bucket_id)
        steps = self.distinct_steps_for_bucket(b)[: max(0, int(n))]
        if not steps:
            return []
        chosen = set(steps)
        popped: list[RetrievalEntry] = []
        seen_steps: set[int] = set()
        survivors: list[RetrievalEntry] = []
        for e in self._buffer:
            if e.bucket_id == b and int(e.step) in chosen:
                if int(e.step) not in seen_steps:
                    seen_steps.add(int(e.step))
                    popped.append(e)
                continue
            survivors.append(e)
        self._buffer = deque(survivors, maxlen=self.max_size)
        self.replayed_total += len(popped)
        return popped

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
        def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.detach().cpu()

        return {
            "max_size": self.max_size,
            "replayed_total": int(self.replayed_total),
            "entries": [
                {
                    "embedding": e.embedding.detach().cpu(),
                    "teacher_id": int(e.teacher_id),
                    "bucket_id": int(e.bucket_id),
                    "step": int(e.step),
                    "teacher_name": str(e.teacher_name),
                    "teacher_output_text": str(e.teacher_output_text),
                    "metadata": dict(e.metadata),
                    "input_ids": _cpu(e.input_ids),
                    "labels": _cpu(e.labels),
                    "kd_row_mask": e.kd_row_mask,
                    "teacher_confidence": e.teacher_confidence,
                }
                for e in self._buffer
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._buffer = deque(maxlen=self.max_size)
        self.replayed_total = int(state.get("replayed_total", 0))
        for raw in state.get("entries", []):
            # ``.get`` on the replay fields: checkpoints written before
            # replay existed simply load as non-replayable entries.
            kd = raw.get("kd_row_mask")
            conf = raw.get("teacher_confidence")
            self._buffer.append(
                RetrievalEntry(
                    embedding=raw["embedding"],
                    teacher_id=int(raw["teacher_id"]),
                    bucket_id=int(raw["bucket_id"]),
                    step=int(raw["step"]),
                    teacher_name=str(raw.get("teacher_name", "")),
                    teacher_output_text=str(raw.get("teacher_output_text", "")),
                    metadata=dict(raw.get("metadata", {})),
                    input_ids=raw.get("input_ids"),
                    labels=raw.get("labels"),
                    kd_row_mask=None if kd is None else float(kd),
                    teacher_confidence=None if conf is None else float(conf),
                )
            )

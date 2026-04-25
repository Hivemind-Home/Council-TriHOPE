"""Teacher logits cache reader.

Layer B carries a ``teacher_logits_path`` like
``teacher_logits/code/train/<uuid>_code.npz``. This module resolves those
paths against a configurable ``root`` (local directory today; an fsspec URL
later) and returns a ``(logits, hit)`` tuple. The collator uses the ``hit``
flag to build a per-row ``teacher_logits_mask`` so KD is skipped on misses
instead of failing the batch.

Expected NPZ schema: a single array under key ``logits`` with shape
``[T, V]`` where ``V`` matches the student vocab. Other keys (e.g. token
ids, top-k indices) are ignored here — swap this file if you move to
sparse top-k storage.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TeacherLogitsCache:
    """LRU-backed NPZ reader keyed by ``(teacher_logits_path)``.

    Returns a ``torch.Tensor`` of shape ``[T, V]`` or ``None`` on miss.
    Missing files are remembered as misses to avoid repeated stat() calls.
    """

    _MISS = object()

    def __init__(self, root: Optional[str], max_items: int = 256) -> None:
        self.root = root
        self.max_items = max_items
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def _resolve(self, path: str) -> Optional[Path]:
        if self.root is None:
            return None
        # Paths in Layer B are relative; absolute paths pass through.
        if os.path.isabs(path):
            full = Path(path)
        else:
            full = Path(self.root) / path
        return full

    def get(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if not path or self.root is None:
            return None

        cached = self._cache.get(path)
        if cached is self._MISS:
            return None
        if cached is not None:
            # LRU touch
            self._cache.move_to_end(path)
            return cached  # type: ignore[return-value]

        resolved = self._resolve(path)
        if resolved is None or not resolved.exists():
            self._remember(path, self._MISS)
            return None

        try:
            with np.load(resolved) as npz:
                arr = npz["logits"] if "logits" in npz.files else npz[npz.files[0]]
        except Exception:
            self._remember(path, self._MISS)
            return None

        tensor = torch.from_numpy(np.asarray(arr)).float()
        self._remember(path, tensor)
        return tensor

    def _remember(self, key: str, value: object) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

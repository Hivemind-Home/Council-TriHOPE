"""Resumable data sampler for bit-exact mid-epoch resume.

``StatefulSampler`` draws a deterministic per-epoch permutation from an
explicit seed (independent of global RNG) and tracks its position, so a
checkpointed run can continue from the exact next batch instead of
restarting a fresh shuffle.

Exactness requires ``num_workers=0`` — worker prefetching advances the
sampler ahead of the batches actually consumed by the training loop.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler


class StatefulSampler(Sampler[int]):
    """Seeded per-epoch permutation with a resumable (epoch, offset) cursor."""

    def __init__(self, data_len: int, seed: int = 42, shuffle: bool = True) -> None:
        if data_len <= 0:
            raise ValueError(f"data_len must be positive, got {data_len}")
        self.data_len = int(data_len)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.offset = 0

    def _permutation(self) -> list[int]:
        if not self.shuffle:
            return list(range(self.data_len))
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.data_len, generator=g).tolist()

    def __iter__(self) -> Iterator[int]:
        if self.offset >= self.data_len:
            # Previous epoch fully consumed — roll to the next one.
            self.epoch += 1
            self.offset = 0
        indices = self._permutation()
        while self.offset < self.data_len:
            idx = indices[self.offset]
            self.offset += 1
            yield idx

    def __len__(self) -> int:
        return self.data_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.offset = 0

    # -- serialisation ----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "data_len": self.data_len,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("data_len", self.data_len)) != self.data_len:
            raise ValueError(
                "StatefulSampler dataset length mismatch: checkpoint has "
                f"{state.get('data_len')}, current dataset has {self.data_len}. "
                "Resuming with a different dataset would silently change the "
                "data order."
            )
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))

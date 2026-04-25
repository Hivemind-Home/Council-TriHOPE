"""Synthetic datasets used by CPU smoke tests and the existing training contract.

Kept byte-equivalent to the pre-refactor behaviour so every existing test that
imports ``SyntheticTeacherSeedData`` or builds a synthetic dataloader keeps
passing. See ``hivemind.data.dataloader.build_dataloader`` for the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic text dataset."""

    vocab_size: int = 256
    seq_len: int = 32
    dataset_size: int = 1000
    seed: int = 42


class SyntheticTextDataset(Dataset):
    """Synthetic text dataset with deterministic random token sequences."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed)
        self.data = torch.randint(
            0,
            config.vocab_size,
            (config.dataset_size, config.seq_len),
            generator=self.generator,
        )

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class SyntheticTeacherSeedData:
    """Seed data for teacher prototype computation, split modularly across K teachers."""

    def __init__(
        self,
        num_teachers: int,
        samples_per_teacher: int,
        vocab_size: int,
        seq_len: int,
        seed: int = 123,
    ) -> None:
        self.num_teachers = num_teachers
        gen = torch.Generator().manual_seed(seed)
        total = num_teachers * samples_per_teacher
        all_data = torch.randint(0, vocab_size, (total, seq_len), generator=gen)

        self.seed_data: dict[str, list[torch.Tensor]] = {}
        for k in range(num_teachers):
            start = k * samples_per_teacher
            end = start + samples_per_teacher
            self.seed_data[f"teacher_{k}"] = [all_data[i] for i in range(start, end)]

    def get_seed_data(self) -> dict[str, list[torch.Tensor]]:
        return self.seed_data

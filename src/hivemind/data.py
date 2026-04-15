"""Dataset implementations for training.

Provides synthetic datasets for development/testing and shard-based datasets
for real training data.
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
    """Synthetic text dataset with random token sequences.

    Generates deterministic random sequences for reproducible smoke tests.
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed)
        # Pre-generate all data for determinism
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
    """Generates synthetic seed data for teacher prototype computation.

    Splits the dataset into K teacher domains by simple modular assignment.
    Each teacher gets every K-th sample starting from its index.
    """

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


def build_dataloader(
    cfg: dict,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader from config.

    Args:
        cfg: data config dict.
        distributed: whether to use DistributedSampler.

    Returns:
        DataLoader instance.
    """
    source = cfg.get("source", "synthetic")

    if source == "synthetic":
        dataset = SyntheticTextDataset(
            SyntheticDataConfig(
                vocab_size=cfg.get("vocab_size", 256),
                seq_len=cfg.get("seq_len", 32),
                dataset_size=cfg.get("dataset_size", 1000),
                seed=cfg.get("seed", 42),
            )
        )
    else:
        raise ValueError(f"Unknown data source: {source}")

    sampler = None
    if distributed:
        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=(sampler is None),
        num_workers=cfg.get("num_workers", 0),
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
    )

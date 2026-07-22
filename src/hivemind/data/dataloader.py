"""DataLoader dispatcher: synthetic vs HuggingFace sources.

The synthetic branch preserves the exact contract used by ``pilot_smoke`` and
the existing test suite. The ``hf`` branch is the new path wired to
``HivemindHFDataset`` and ``DistillCollator`` for real Layer A/B supervision.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .synthetic import SyntheticDataConfig, SyntheticTextDataset


def _build_synthetic(cfg: dict) -> Dataset:
    return SyntheticTextDataset(
        SyntheticDataConfig(
            vocab_size=cfg.get("vocab_size", 256),
            seq_len=cfg.get("seq_len", 32),
            dataset_size=cfg.get("dataset_size", 1000),
            seed=cfg.get("seed", 42),
        )
    )


def _build_hf(cfg: dict) -> tuple[Dataset, Any]:
    # Lazy imports: the ``data`` extra may not be installed in CPU-smoke envs.
    from .collate import DistillCollator
    from .hf_loader import HivemindHFConfig, HivemindHFDataset
    from .teacher_cache import TeacherLogitsCache
    from .tokenizer import build_tokenizer

    hf_cfg = HivemindHFConfig(
        domains=list(cfg.get("domains", ["code"])),
        repo_prefix=cfg.get("repo_prefix", "hivemind-research"),
        split=cfg.get("split", "train"),
        tokenizer_name=cfg["tokenizer"],
        max_seq_len=int(cfg.get("max_seq_len", 2048)),
        teacher_logits_root=cfg.get("teacher_logits_root"),
        cache_dir=cfg.get("hf_cache_dir"),
        streaming=bool(cfg.get("streaming", False)),
        max_rows_per_domain=cfg.get("max_rows_per_domain"),
    )

    tokenizer = build_tokenizer(hf_cfg.tokenizer_name)
    dataset = HivemindHFDataset(hf_cfg)
    logits_cache = TeacherLogitsCache(
        root=hf_cfg.teacher_logits_root,
        max_items=int(cfg.get("logits_cache_size", 256)),
    )
    collator = DistillCollator(
        tokenizer=tokenizer,
        max_seq_len=hf_cfg.max_seq_len,
        logits_cache=logits_cache,
    )
    return dataset, collator


def build_dataloader(
    cfg: dict,
    distributed: bool = False,
    stateful: bool = False,
):
    """Build a DataLoader from config.

    ``cfg["source"]`` in ``{"synthetic", "hf"}`` selects the path. Synthetic
    batches are plain ``torch.Tensor[B, T]``; HF batches are dicts — see
    ``DistillCollator`` for the schema.

    With ``stateful=True`` (the training path) the loader is driven by a
    :class:`StatefulSampler` whose (epoch, offset) cursor is checkpointable
    for bit-exact mid-epoch resume, and the return value is
    ``(loader, sampler)``. Evaluation loaders should keep the default so
    every eval pass sees the same batches.
    """
    source = cfg.get("source", "synthetic")
    collate_fn = None

    if source == "synthetic":
        dataset: Dataset = _build_synthetic(cfg)
    elif source == "hf":
        dataset, collate_fn = _build_hf(cfg)
    else:
        raise ValueError(f"Unknown data source: {source}")

    if distributed:
        sampler = DistributedSampler(dataset)
        stateful_sampler = None
    elif stateful:
        from .sampler import StatefulSampler

        stateful_sampler = StatefulSampler(
            data_len=len(dataset),  # type: ignore[arg-type]
            seed=int(cfg.get("shuffle_seed", cfg.get("seed", 42))),
            shuffle=bool(cfg.get("shuffle", True)),
        )
        sampler = stateful_sampler
    else:
        sampler = None
        stateful_sampler = None

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=(sampler is None and not distributed and cfg.get("shuffle", True)),
        num_workers=cfg.get("num_workers", 0),
        sampler=sampler,
        drop_last=cfg.get("drop_last", True),
        pin_memory=cfg.get("pin_memory", True),
        collate_fn=collate_fn,
    )
    if stateful:
        return loader, stateful_sampler
    return loader

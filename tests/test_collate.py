"""Tests for DistillCollator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.data.collate import IGNORE_INDEX, DistillCollator
from hivemind.data.teacher_cache import TeacherLogitsCache


def test_basic_batch_shapes(whitespace_tokenizer, hf_sample_rows):
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=32, logits_cache=cache
    )

    batch = collator(hf_sample_rows[:4])

    assert batch["input_ids"].shape[0] == 4
    T = batch["input_ids"].shape[1]
    assert batch["attention_mask"].shape == (4, T)
    assert batch["labels"].shape == (4, T)
    assert batch["teacher_logits"].shape == (4, T, whitespace_tokenizer.vocab_size)
    assert batch["teacher_logits_mask"].shape == (4,)
    assert batch["teacher_logits_mask"].sum().item() == 0  # no cache root


def test_labels_mask_prompt_and_pad(whitespace_tokenizer, hf_sample_rows):
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=24, logits_cache=cache,
        prompt_max_fraction=0.5,
    )

    batch = collator(hf_sample_rows[:2])

    labels = batch["labels"]
    attn = batch["attention_mask"]
    input_ids = batch["input_ids"]

    for b in range(labels.shape[0]):
        # Every -100 position is either a prompt token or pad; never a
        # response token — we check that at least one non-ignored label
        # exists (otherwise CE would be undefined).
        assert (labels[b] != IGNORE_INDEX).any()
        # pad positions: attention_mask = 0 → labels must be ignored
        pad_mask = attn[b] == 0
        assert (labels[b][pad_mask] == IGNORE_INDEX).all()


def test_kd_mask_set_when_npz_hits(whitespace_tokenizer, hf_sample_rows, npz_cache_root):
    # npz_cache_root fixture writes logits with shape [8, 256] for the
    # first six sample_ids — matches vocab_size on the tokenizer.
    cache = TeacherLogitsCache(root=str(npz_cache_root))
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=48, logits_cache=cache
    )

    batch = collator(hf_sample_rows[:6])
    # At least one row should be KD-active.
    assert batch["teacher_logits_mask"].sum().item() >= 1


def test_gold_fallback_when_no_teacher_text(whitespace_tokenizer):
    rows = [{
        "sample_id": "only-gold",
        "input_text": "translate hello",
        "target_text": "bonjour",
        "teacher_id": "mystery",
        "domain": "misc",
    }]
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=16, logits_cache=cache
    )

    batch = collator(rows)

    assert batch["metadata"]["label_source"][0] == "gold"
    assert batch["teacher_logits_mask"].item() == 0.0


def test_raises_on_all_empty_rows(whitespace_tokenizer):
    rows = [{"sample_id": "x"}, {"sample_id": "y"}]
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=8, logits_cache=cache
    )
    with pytest.raises(ValueError):
        collator(rows)

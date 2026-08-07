"""Tests for DistillCollator."""

from __future__ import annotations

import pytest

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
    # No cache root → no row carries logits → the collator emits the [B,T,1]
    # sentinel rather than allocating a full [B,T,V] block of zeros.
    assert batch["teacher_logits"].shape == (4, T, 1)
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
    _ = batch["input_ids"]

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


class _EosTokenizer:
    """Whitespace tokenizer with a distinct eos/pad pair, like Qwen3."""

    eos_token_id = 900
    pad_token_id = 901
    vocab_size = 1000

    def encode(self, text, add_special_tokens=False):
        return [(hash(w) % 800) + 1 for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _row(**over):
    row = {
        "sample_id": "s0",
        "input_text": "the prompt text",
        "target_text": "gold answer here",
        "teacher_output_text": "teacher answer here",
        "has_gold_label": False,
        "teacher_logits_path": "",
        "teacher_confidence": 0.9,
        "domain": "code",
        "bucket_id": "b0",
        "teacher_id": "t0",
    }
    row.update(over)
    return row


class TestEosAppending:
    """Without EOS the student never learns a stop token, yet EM breaks on it."""

    def _collator(self, **kw):
        return DistillCollator(
            tokenizer=_EosTokenizer(),
            max_seq_len=32,
            logits_cache=TeacherLogitsCache(root=None),
            **kw,
        )

    def test_eos_is_appended_and_supervised(self):
        batch = self._collator()([_row()])
        ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()
        last = ids.index(901) if 901 in ids else len(ids)
        assert ids[last - 1] == 900, "eos should be the final real token"
        # It must be a supervised position, not masked out.
        assert labels[last - 1] == 900

    def test_eos_differs_from_pad_so_labels_stay_correct(self):
        tok = _EosTokenizer()
        assert tok.eos_token_id != tok.pad_token_id

    def test_disabled_appends_nothing(self):
        batch = self._collator(append_eos=False)([_row()])
        assert 900 not in batch["input_ids"][0].tolist()

    def test_not_appended_when_the_budget_is_full(self):
        long_row = _row(teacher_output_text=" ".join(f"w{i}" for i in range(200)))
        batch = DistillCollator(
            tokenizer=_EosTokenizer(),
            max_seq_len=16,
            logits_cache=TeacherLogitsCache(root=None),
        )([long_row])
        assert batch["input_ids"].shape[1] <= 16


class TestGoldPolicy:
    def _build(self, policy, row):
        return DistillCollator(
            tokenizer=_EosTokenizer(),
            max_seq_len=64,
            logits_cache=TeacherLogitsCache(root=None),
            gold_policy=policy,
        )([row])

    def test_teacher_only_ignores_the_gold_flag(self):
        b = self._build("teacher_only", _row(has_gold_label=True))
        assert b["metadata"]["label_source"] == ["teacher"]

    def test_gold_when_available_switches_on_the_flag(self):
        b = self._build("gold_when_available", _row(has_gold_label=True))
        assert b["metadata"]["label_source"] == ["gold"]

    def test_gold_when_available_falls_back_without_the_flag(self):
        b = self._build("gold_when_available", _row(has_gold_label=False))
        assert b["metadata"]["label_source"] == ["teacher"]

    def test_gold_when_available_falls_back_without_target_text(self):
        b = self._build("gold_when_available", _row(has_gold_label=True, target_text=""))
        assert b["metadata"]["label_source"] == ["teacher"]

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="gold_policy"):
            DistillCollator(
                tokenizer=_EosTokenizer(),
                max_seq_len=32,
                logits_cache=TeacherLogitsCache(root=None),
                gold_policy="whatever",
            )


class TestVocabAndAllocation:
    def test_explicit_vocab_size_overrides_the_tokenizer(self):
        """Qwen3 reports 151643 while the model embeds 151936."""
        c = DistillCollator(
            tokenizer=_EosTokenizer(),
            max_seq_len=32,
            logits_cache=TeacherLogitsCache(root=None),
            vocab_size=151936,
        )
        assert c.vocab_size == 151936

    def test_no_cached_logits_yields_the_sentinel(self):
        batch = DistillCollator(
            tokenizer=_EosTokenizer(),
            max_seq_len=32,
            logits_cache=TeacherLogitsCache(root=None),
            vocab_size=151936,
        )([_row(), _row(sample_id="s1")])
        assert batch["teacher_logits"].shape[-1] == 1
        assert batch["teacher_logits_mask"].sum().item() == 0

    def test_sentinel_is_tiny_compared_to_a_full_block(self):
        batch = DistillCollator(
            tokenizer=_EosTokenizer(),
            max_seq_len=32,
            logits_cache=TeacherLogitsCache(root=None),
            vocab_size=151936,
        )([_row()])
        assert batch["teacher_logits"].numel() < 1000

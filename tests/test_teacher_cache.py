"""Tests for the NPZ teacher logits cache."""

from __future__ import annotations

import numpy as np
import torch

from hivemind.data.teacher_cache import TeacherLogitsCache


def test_hit_returns_tensor(tmp_path):
    path = tmp_path / "sub" / "x.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, logits=np.random.randn(4, 8).astype(np.float32))

    cache = TeacherLogitsCache(root=str(tmp_path))
    out = cache.get("sub/x.npz")

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 8)
    assert out.dtype == torch.float32


def test_miss_returns_none_and_is_remembered(tmp_path):
    cache = TeacherLogitsCache(root=str(tmp_path))
    assert cache.get("missing.npz") is None
    # Second call must still return None without re-stat'ing the file
    assert cache.get("missing.npz") is None
    assert cache.size == 1  # miss sentinel occupies one slot


def test_none_root_disables_cache():
    cache = TeacherLogitsCache(root=None)
    assert cache.get("anything.npz") is None


def test_lru_eviction(tmp_path):
    for i in range(5):
        p = tmp_path / f"{i}.npz"
        np.savez(p, logits=np.ones((1, 1), dtype=np.float32))

    cache = TeacherLogitsCache(root=str(tmp_path), max_items=3)
    for i in range(5):
        cache.get(f"{i}.npz")

    assert cache.size == 3
    # Earliest paths (0, 1) should have been evicted
    # Re-fetching rehydrates from disk
    out = cache.get("0.npz")
    assert out is not None


def test_falls_back_to_first_key_if_no_logits_key(tmp_path):
    p = tmp_path / "alt.npz"
    np.savez(p, foo=np.ones((2, 3), dtype=np.float32))

    cache = TeacherLogitsCache(root=str(tmp_path))
    out = cache.get("alt.npz")
    assert out is not None
    assert out.shape == (2, 3)

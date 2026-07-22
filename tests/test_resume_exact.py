"""Bit-exact resume: N steps + checkpoint + resume ≡ 2N uninterrupted steps.

Also unit-tests StatefulSampler determinism/cursor and the deterministic
bucket hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from hivemind.data.sampler import StatefulSampler
from hivemind.training import _bucket_to_int, run_training_loop


class TestBucketHash:
    def test_deterministic_known_value(self) -> None:
        # Hardcoded expected value: catches any regression to PYTHONHASHSEED-
        # dependent hashing (which changes per process).
        assert _bucket_to_int("math_algebra_medium", 0) == 2094931850
        assert _bucket_to_int("math_algebra_medium", 0) == _bucket_to_int(
            "math_algebra_medium", 99
        )

    def test_fallback(self) -> None:
        assert _bucket_to_int(None, 7) == 7
        assert _bucket_to_int("", 3) == 3


class TestStatefulSampler:
    def test_same_seed_same_order(self) -> None:
        a = list(iter(StatefulSampler(20, seed=5)))
        b = list(iter(StatefulSampler(20, seed=5)))
        assert a == b
        assert sorted(a) == list(range(20))

    def test_epochs_differ(self) -> None:
        s = StatefulSampler(20, seed=5)
        first = list(iter(s))
        second = list(iter(s))  # auto-advances to epoch 1
        assert first != second
        assert s.epoch == 1

    def test_midepoch_resume_continues(self) -> None:
        s = StatefulSampler(10, seed=1)
        it = iter(s)
        consumed = [next(it) for _ in range(4)]
        state = s.state_dict()

        fresh = StatefulSampler(10, seed=999)  # seed overwritten by load
        fresh.load_state_dict(state)
        rest = list(iter(fresh))
        full = list(iter(StatefulSampler(10, seed=1)))
        assert consumed + rest == full

    def test_length_mismatch_raises(self) -> None:
        s = StatefulSampler(10, seed=1)
        with pytest.raises(ValueError):
            StatefulSampler(11, seed=1).load_state_dict(s.state_dict())

    def test_no_shuffle_sequential(self) -> None:
        s = StatefulSampler(5, seed=0, shuffle=False)
        assert list(iter(s)) == [0, 1, 2, 3, 4]


def _cfg(tmp_path: Path, steps: int, resume_from: str | None = None) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "vocab_size": 64,
                "dim": 32,
                "num_layers": 2,
                "heads": 4,
                "max_seq_len": 32,
                "lora": {"rank": 4, "alpha": 8.0},
            },
            "teachers": {"num_teachers": 2, "mode": "synthetic"},
            "controller": {"consolidation": {"period": 0}},
            "data": {
                "source": "synthetic",
                # Small dataset so the run crosses an epoch boundary — resume
                # must restore the mid-epoch cursor, not restart the shuffle.
                "dataset_size": 20,
                "seq_len": 16,
                "vocab_size": 64,
                "batch_size": 4,
            },
            "train": {"steps": steps, "device": "cpu", "seed": 123, "log_interval": 1},
            "optim": {"lr": 1e-3},
            "checkpoint": {
                "enabled": True,
                "dir": str(tmp_path / "ckpt"),
                "save_every": 10_000,  # only the final-step save fires
                "keep_last": 5,
                "resume_from": resume_from,
            },
            "logging": {
                "enabled": True,
                "backend": "json",
                "path": str(tmp_path / "metrics.jsonl"),
                "events_path": str(tmp_path / "events.jsonl"),
            },
        }
    )


def _read_losses(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text().strip().splitlines():
        rec = json.loads(line)
        if "loss/total" in rec:
            out[rec["step"]] = rec["loss/total"]
    return out


class TestBitExactResume:
    def test_resume_matches_uninterrupted(self, tmp_path: Path) -> None:
        n, total = 8, 16
        device = torch.device("cpu")

        # Reference: uninterrupted 2N steps.
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        run_training_loop(_cfg(ref_dir, steps=total), device=device)
        ref_losses = _read_losses(ref_dir / "metrics.jsonl")

        # Interrupted: N steps, checkpoint at N-1, then resume to 2N.
        part_dir = tmp_path / "part"
        part_dir.mkdir()
        run_training_loop(_cfg(part_dir, steps=n), device=device)
        assert (part_dir / "ckpt" / f"step_{n - 1:08d}").exists()

        run_training_loop(
            _cfg(part_dir, steps=total, resume_from="latest"), device=device
        )
        resumed_losses = _read_losses(part_dir / "metrics.jsonl")

        assert set(resumed_losses) == set(ref_losses)
        for step in sorted(ref_losses):
            assert resumed_losses[step] == pytest.approx(
                ref_losses[step], abs=0.0
            ), f"loss diverged at step {step}"

        # The resumed run's second-half events continue where the first left
        # off (a `resume` marker separates the segments).
        events = [
            json.loads(ln)
            for ln in (part_dir / "events.jsonl").read_text().strip().splitlines()
        ]
        assert any(e["type"] == "resume" and e["step"] == n for e in events)

    def test_resumed_checkpoint_state_matches_reference(self, tmp_path: Path) -> None:
        n, total = 6, 12
        device = torch.device("cpu")

        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        run_training_loop(_cfg(ref_dir, steps=total), device=device)

        part_dir = tmp_path / "part"
        part_dir.mkdir()
        run_training_loop(_cfg(part_dir, steps=n), device=device)
        run_training_loop(
            _cfg(part_dir, steps=total, resume_from="latest"), device=device
        )

        final = f"step_{total - 1:08d}"
        for fname in ("base.pt", "lora.pt"):
            ref_sd = torch.load(
                ref_dir / "ckpt" / final / fname, weights_only=False
            )
            res_sd = torch.load(
                part_dir / "ckpt" / final / fname, weights_only=False
            )
            assert set(ref_sd) == set(res_sd)
            for key in ref_sd:
                assert torch.equal(ref_sd[key], res_sd[key]), f"{fname}:{key} differs"

        ref_data = torch.load(ref_dir / "ckpt" / final / "data.pt", weights_only=False)
        res_data = torch.load(part_dir / "ckpt" / final / "data.pt", weights_only=False)
        assert ref_data == res_data

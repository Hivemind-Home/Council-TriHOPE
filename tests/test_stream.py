"""Tests for the phased continual stream schedule."""

from __future__ import annotations

import pytest

from hivemind.data.stream import (
    PhaseSpec,
    RecurrenceSpec,
    StreamBatchSampler,
    StreamConfig,
    StreamSchedule,
    parse_stream_config,
)


class _FakeDataset:
    """Two domains: 'a' (indices 0..59, 6 buckets of 10) and 'b' (60..99)."""

    def indices_for_domain(self, name):
        return range(0, 60) if name == "a" else range(60, 100)

    def indices_by_bucket(self, name):
        if name == "a":
            return {f"a_bucket_{k}": list(range(k * 10, (k + 1) * 10)) for k in range(6)}
        return {f"b_bucket_{k}": list(range(60 + k * 10, 60 + (k + 1) * 10)) for k in range(4)}

    def gold_indices(self, name):
        return list(self.indices_for_domain(name))[:5]


def _schedule(phases: list[PhaseSpec], batch_size: int = 2, seed: int = 7) -> StreamSchedule:
    return StreamSchedule(
        StreamConfig(enabled=True, phases=phases),
        _FakeDataset(),
        batch_size=batch_size,
        seed=seed,
    )


class TestDeterminism:
    def test_same_seed_same_sequence(self) -> None:
        phases = [
            PhaseSpec(name="warm", steps=5, mode="random", domain="a"),
            PhaseSpec(
                name="rec", steps=12, mode="recurrent", domain="a",
                recurrence=RecurrenceSpec(num_buckets=3, samples_per_bucket=4),
            ),
        ]
        s1 = _schedule(phases)
        s2 = _schedule(phases)
        for t in range(s1.total_steps):
            assert s1.indices_for_step(t) == s2.indices_for_step(t)
        assert s1.config_digest() == s2.config_digest()

    def test_different_seed_differs(self) -> None:
        phases = [PhaseSpec(name="warm", steps=10, mode="random", domain="a")]
        s1 = _schedule(phases, seed=1)
        s2 = _schedule(phases, seed=2)
        assert any(
            s1.indices_for_step(t) != s2.indices_for_step(t) for t in range(10)
        )
        assert s1.config_digest() != s2.config_digest()


class TestPhases:
    def test_boundaries_and_lookup(self) -> None:
        phases = [
            PhaseSpec(name="p1", steps=4, mode="random", domain="a"),
            PhaseSpec(name="p2", steps=6, mode="random", domain="b"),
        ]
        s = _schedule(phases)
        assert s.total_steps == 10
        assert s.phase_boundaries == [(0, "p1"), (4, "p2")]
        assert s.phase_at(0).name == "p1"
        assert s.phase_at(3).name == "p1"
        assert s.phase_at(4).name == "p2"
        assert s.phase_at(9).name == "p2"

    def test_domain_purity(self) -> None:
        phases = [
            PhaseSpec(name="p1", steps=4, mode="random", domain="a"),
            PhaseSpec(name="p2", steps=4, mode="random", domain="b"),
        ]
        s = _schedule(phases)
        for t in range(4):
            assert all(i < 60 for i in s.indices_for_step(t))
        for t in range(4, 8):
            assert all(i >= 60 for i in s.indices_for_step(t))


class TestRecurrent:
    def test_bucket_pure_batches_and_recurrence(self) -> None:
        rec = RecurrenceSpec(num_buckets=3, samples_per_bucket=4)
        phases = [
            PhaseSpec(name="rec", steps=30, mode="recurrent", domain="a", recurrence=rec)
        ]
        s = _schedule(phases)
        buckets = s.phase_buckets("rec")
        assert len(buckets) == 3
        subsets = {k: set(v) for k, v in buckets.items()}
        # Default period == num_buckets: every step is a bucket step and each
        # batch stays inside exactly one bucket's fixed subset.
        seen_count: dict[str, int] = {k: 0 for k in subsets}
        for t in range(30):
            batch = set(s.indices_for_step(t))
            owners = [k for k, sub in subsets.items() if batch <= sub]
            assert len(owners) == 1, f"step {t} not bucket-pure"
            seen_count[owners[0]] += 1
        assert all(c == 10 for c in seen_count.values())  # 30 steps / 3 buckets

    def test_reuse_from_replays_same_buckets(self) -> None:
        rec = RecurrenceSpec(num_buckets=2, samples_per_bucket=4)
        phases = [
            PhaseSpec(name="first", steps=6, mode="recurrent", domain="a", recurrence=rec),
            PhaseSpec(name="other", steps=4, mode="random", domain="b"),
            PhaseSpec(
                name="revisit", steps=6, mode="recurrent", domain="a",
                recurrence=RecurrenceSpec(num_buckets=2, samples_per_bucket=4, reuse_from="first"),
            ),
        ]
        s = _schedule(phases)
        assert s.phase_buckets("revisit") == s.phase_buckets("first")

    def test_reuse_from_later_phase_raises(self) -> None:
        phases = [
            PhaseSpec(
                name="revisit", steps=4, mode="recurrent", domain="a",
                recurrence=RecurrenceSpec(reuse_from="nonexistent"),
            ),
        ]
        with pytest.raises(ValueError, match="reuse_from"):
            _schedule(phases)

    def test_insufficient_buckets_raises(self) -> None:
        phases = [
            PhaseSpec(
                name="rec", steps=4, mode="recurrent", domain="b",
                recurrence=RecurrenceSpec(num_buckets=10, samples_per_bucket=4),
            ),
        ]
        with pytest.raises(ValueError, match="buckets"):
            _schedule(phases)


class TestNovel:
    def test_unique_single_use_rows_with_overrides(self) -> None:
        phases = [
            PhaseSpec(name="warm", steps=3, mode="random", domain="a"),
            PhaseSpec(name="nov", steps=5, mode="novel", domain="b"),
        ]
        s = _schedule(phases)
        novel_rows: list[int] = []
        for t in range(3, 8):
            batch = s.indices_for_step(t)
            novel_rows.extend(batch)
            assert s.bucket_override(t) is not None
            assert s.bucket_override(t).startswith("novel_nov_")
        # Each novel row is used exactly once.
        assert len(novel_rows) == len(set(novel_rows))
        # Unique override per step.
        overrides = {s.bucket_override(t) for t in range(3, 8)}
        assert len(overrides) == 5
        # Non-novel steps have no override.
        assert s.bucket_override(0) is None

    def test_pool_exhaustion_raises(self) -> None:
        phases = [
            PhaseSpec(name="nov", steps=100, mode="novel", domain="b"),  # 40 rows only
        ]
        with pytest.raises(ValueError, match="novel"):
            _schedule(phases, batch_size=2)


class TestMixed:
    def test_weighted_domains(self) -> None:
        phases = [
            PhaseSpec(
                name="mix", steps=50, mode="mixed",
                domains=["a", "b"], weights=[0.5, 0.5],
            )
        ]
        s = _schedule(phases)
        domain_a_steps = sum(
            1 for t in range(50) if all(i < 60 for i in s.indices_for_step(t))
        )
        domain_b_steps = sum(
            1 for t in range(50) if all(i >= 60 for i in s.indices_for_step(t))
        )
        assert domain_a_steps + domain_b_steps == 50  # every batch domain-pure
        assert 10 <= domain_a_steps <= 40  # roughly balanced


class TestSamplerAndParsing:
    def test_batch_sampler_start_step(self) -> None:
        phases = [PhaseSpec(name="p", steps=6, mode="random", domain="a")]
        s = _schedule(phases)
        full = list(StreamBatchSampler(s))
        tail = list(StreamBatchSampler(s, start_step=4))
        assert full[4:] == tail
        assert len(tail) == 2

    def test_parse_from_dict(self) -> None:
        cfg = parse_stream_config(
            {
                "enabled": True,
                "phases": [
                    {"name": "w", "steps": 3, "mode": "random", "domain": "a"},
                    {
                        "name": "r", "steps": 4, "mode": "recurrent", "domain": "a",
                        "recurrence": {"num_buckets": 2, "samples_per_bucket": 4},
                    },
                ],
            }
        )
        assert cfg.enabled
        assert len(cfg.phases) == 2
        assert cfg.phases[1].recurrence.num_buckets == 2

    def test_parse_empty_disabled(self) -> None:
        assert parse_stream_config(None).enabled is False
        assert parse_stream_config({}).enabled is False

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            PhaseSpec(name="x", steps=1, mode="bogus", domain="a")

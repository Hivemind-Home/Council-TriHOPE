"""Corrupted-teacher stream (task T4, experiment E5)."""

from __future__ import annotations

import pytest
from test_stream import _FakeDataset

from hivemind.data.corruption import (
    CorruptedTeacherView,
    CorruptionPlan,
    CorruptionSpec,
    degrade_text,
    parse_corruption_config,
)
from hivemind.data.preflight import validate
from hivemind.data.stream import PhaseSpec, RecurrenceSpec, StreamConfig, StreamSchedule

PHASES = [
    PhaseSpec(name="warm", steps=6, mode="random", domain="b"),  # other domain: no overlap
    PhaseSpec(
        name="a_recurrent", steps=40, mode="recurrent", domain="a",
        recurrence=RecurrenceSpec(num_buckets=4, samples_per_bucket=4, revisit_period=5,
                                  novel_fraction=0.5),
    ),
    PhaseSpec(name="tail", steps=10, mode="mixed", domains=["a", "b"]),
]


def _spec(**kw) -> CorruptionSpec:
    base = dict(enabled=True, domain="a", phase="a_recurrent", fraction=0.5, mode="shuffle",
                tag="a_teacher_corrupted")
    base.update(kw)
    return CorruptionSpec(**base)


def _schedule(corruption=None, seed: int = 7) -> StreamSchedule:
    return StreamSchedule(
        StreamConfig(enabled=True, phases=PHASES), _FakeDataset(), batch_size=2, seed=seed,
        corruption=corruption,
    )


def _phase_steps(s: StreamSchedule, name: str) -> range:
    starts = dict((n, st) for st, n in s.phase_boundaries)
    steps = {p.name: p.steps for p in PHASES}
    return range(starts[name], starts[name] + steps[name])


class TestSelection:
    def test_step_fraction_exact_within_bucket_granularity(self) -> None:
        s = _schedule(_spec())
        steps = _phase_steps(s, "a_recurrent")
        corrupted_steps = [t for t in steps if s.is_corrupted_step(t)]
        # 4 buckets × 8 visits = 32 bucket steps (2 of 4 buckets corrupted → 16)
        # + 8 other steps (4 corrupted) = 20 of 40.
        assert len(corrupted_steps) == 20
        assert abs(len(corrupted_steps) / len(steps) - 0.5) <= 1 / 4

    def test_every_batch_is_teacher_pure(self) -> None:
        s = _schedule(_spec())
        for t in _phase_steps(s, "a_recurrent"):
            flags = {i in s.corrupted_indices for i in s.indices_for_step(t)}
            assert len(flags) == 1

    def test_only_rows_of_the_named_phase(self) -> None:
        s = _schedule(_spec())
        served = {i for t in _phase_steps(s, "a_recurrent") for i in s.indices_for_step(t)}
        assert s.corrupted_indices and s.corrupted_indices <= served
        assert s.corrupted_rows_by_phase["warm"] == 0
        assert s.corrupted_rows_by_phase["a_recurrent"] == 40  # 20 steps × 2 rows
        # A corrupted background row may be re-drawn by the mixed tail: reported.
        assert s.corrupted_rows_by_phase["tail"] >= 0

    def test_same_seed_same_set_different_seed_differs(self) -> None:
        a, b = _schedule(_spec(), seed=3), _schedule(_spec(), seed=3)
        assert a.corrupted_indices == b.corrupted_indices
        assert a.corruption_partner == b.corruption_partner
        c = _schedule(_spec(), seed=4)
        assert a.corrupted_indices != c.corrupted_indices

    def test_digest_changes_only_when_enabled(self) -> None:
        clean = _schedule(None).config_digest()
        assert _schedule(CorruptionSpec(enabled=False, domain="a", phase="a_recurrent")
                         ).config_digest() == clean
        on = _schedule(_spec()).config_digest()
        assert on != clean
        assert _schedule(_spec(fraction=0.25)).config_digest() != on
        assert _schedule(_spec(mode="degrade")).config_digest() != on

    def test_shuffle_partner_is_a_derangement(self) -> None:
        s = _schedule(_spec())
        partner = s.corruption_partner
        assert set(partner) == s.corrupted_indices
        assert all(partner[i] != i for i in partner)
        assert set(partner.values()) == s.corrupted_indices  # a bijection

    def test_bad_phase_or_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="phase"):
            _schedule(_spec(phase="nope"))
        with pytest.raises(ValueError, match="domain"):
            _schedule(_spec(domain="b"))

    def test_invalid_spec_values(self) -> None:
        with pytest.raises(ValueError):
            CorruptionSpec(mode="flip")
        with pytest.raises(ValueError):
            CorruptionSpec(fraction=1.5)
        assert parse_corruption_config(None).enabled is False


class _Rows:
    def __init__(self, n: int = 6) -> None:
        self.rows = [
            {"input_text": f"q{i}", "teacher_output_text": f"answer {i} is {i * 10}",
             "teacher_id": "a_teacher", "teacher_confidence": 0.9, "domain": "a",
             "teacher_logits_path": f"/cache/{i}.npz"}
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return dict(self.rows[i])

    def indices_for_domain(self, name):  # delegated accessor
        return range(len(self.rows))


class TestView:
    def test_rewrites_only_corrupted_rows(self) -> None:
        ds = _Rows()
        plan = CorruptionPlan(spec=_spec(confidence=0.2), corrupted={1, 3}, partner_of={1: 3, 3: 1})
        view = CorruptedTeacherView(ds, plan)
        assert len(view) == 6 and list(view.indices_for_domain("a")) == list(range(6))
        clean = view[0]
        assert clean == ds[0]
        bad = view[1]
        assert bad["teacher_output_text"] == ds[3]["teacher_output_text"]
        assert bad["teacher_id"] == "a_teacher_corrupted"
        assert bad["teacher_confidence"] == 0.2
        assert bad["teacher_logits_path"] is None and bad["corrupted"] is True
        assert view[3]["teacher_output_text"] == ds[1]["teacher_output_text"]

    def test_confidence_none_keeps_the_rows_own(self) -> None:
        plan = CorruptionPlan(spec=_spec(mode="degrade"), corrupted={2}, partner_of={})
        bad = CorruptedTeacherView(_Rows(), plan)[2]
        assert bad["teacher_confidence"] == 0.9
        assert bad["teacher_output_text"].endswith("Final answer: 21")

    def test_shuffle_without_partner_falls_back_to_degrade(self) -> None:
        plan = CorruptionPlan(spec=_spec(), corrupted={2}, partner_of={})
        assert "Final answer: 21" in CorruptedTeacherView(_Rows(), plan)[2]["teacher_output_text"]


class TestDegrade:
    def test_last_number_is_incremented_and_text_truncated(self) -> None:
        out = degrade_text("step one two three four five six seven eight = 41")
        assert out.endswith("Final answer: 42")
        assert out.startswith("step one")
        assert "eight" not in out

    def test_decimal_and_no_number(self) -> None:
        assert degrade_text("x = 2.5").endswith("Final answer: 3.5")
        assert degrade_text("no digits here at all").endswith("Final answer: 0")


class TestRouterAndPreflight:
    def test_domain_to_index_first_wins_with_extra_teacher(self) -> None:
        import torch

        from hivemind.teacher_registry import create_hf_cache_teachers
        from hivemind.teacher_router import MetadataRouter

        teachers = create_hf_cache_teachers(
            domains=["a", "b"], teacher_ids={"a": "a_teacher", "b": "b_teacher"},
            vocab_size=16, device=torch.device("cpu"),
            extra=[("a_teacher_corrupted", "a")],
        )
        assert [t.name for t in teachers] == ["a_teacher", "b_teacher", "a_teacher_corrupted"]
        d2i: dict[str, int] = {}
        for i, t in enumerate(teachers):
            d2i.setdefault(t.model.domain, i)
        router = MetadataRouter([t.name for t in teachers], d2i)
        idx = router.route(
            {"teacher_id": ["a_teacher_corrupted", "unknown", "b_teacher"],
             "domain": ["a", "a", "b"]},
            device=torch.device("cpu"),
        )
        assert idx.tolist() == [2, 0, 1]

    def test_preflight_validates_the_block(self) -> None:
        data_cfg = {"domains": ["a", "b"], "corrupt_teacher": {
            "enabled": True, "domain": "a", "phase": "a_recurrent", "fraction": 0.5,
            "mode": "shuffle", "tag": "a_teacher_corrupted"}}
        stream_cfg = {"phases": [{"name": "a_recurrent", "domain": "a", "mode": "recurrent"}]}
        rep = validate(data_cfg, stream_cfg, {"teacher_ids": {"a": "a_teacher"}})
        assert not [e for e in rep.errors if "corrupt" in e]
        assert rep.facts["corrupt_teacher"]["tag"] == "a_teacher_corrupted"

        bad = dict(data_cfg, corrupt_teacher={**data_cfg["corrupt_teacher"], "phase": "x"})
        rep = validate(bad, stream_cfg, {"teacher_ids": {"a": "a_teacher"}})
        assert any("corrupt_teacher.phase" in e for e in rep.errors)
        collide = dict(data_cfg, corrupt_teacher={**data_cfg["corrupt_teacher"],
                                                 "tag": "a_teacher"})
        rep = validate(collide, stream_cfg, {"teacher_ids": {"a": "a_teacher"}})
        assert any("collides" in e for e in rep.errors)

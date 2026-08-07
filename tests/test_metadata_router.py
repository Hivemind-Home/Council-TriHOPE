"""Tests for MetadataRouter."""

from __future__ import annotations

import pytest
import torch

from hivemind.teacher_router import MetadataRouter


def test_route_by_teacher_id():
    router = MetadataRouter(
        teacher_names=["code_teacher_deepseek_r1", "math_teacher_deepseek_r1_1p5b"]
    )
    meta = {
        "teacher_id": [
            "code_teacher_deepseek_r1",
            "math_teacher_deepseek_r1_1p5b",
            "code_teacher_deepseek_r1",
        ],
        "domain": ["code", "math", "code"],
    }
    idx = router.route(meta, device=torch.device("cpu"))
    assert idx.tolist() == [0, 1, 0]
    assert router.miss_count == 0


def test_domain_fallback_when_teacher_id_unknown():
    router = MetadataRouter(
        teacher_names=["code_t", "math_t"],
        domain_to_index={"code": 0, "math": 1},
    )
    meta = {"teacher_id": ["", ""], "domain": ["math", "code"]}
    idx = router.route(meta, device=torch.device("cpu"))
    assert idx.tolist() == [1, 0]


def test_unknown_falls_back_and_counts_miss():
    router = MetadataRouter(teacher_names=["a", "b"])
    meta = {"teacher_id": ["ghost"], "domain": ["elsewhere"]}
    idx = router.route(meta, device=torch.device("cpu"))
    assert idx.tolist() == [0]
    assert router.miss_count == 1


class TestMissRate:
    """A 100% miss rate still trains in cache mode — it must be reportable."""

    def test_miss_rate_tracks_routed_samples(self):
        router = MetadataRouter(teacher_names=["a"], domain_to_index={"a": 0})
        meta = {"teacher_id": ["a", "?", "?", "?"], "domain": ["a", "?", "?", "?"]}
        router.route(meta, device=torch.device("cpu"))
        assert router.routed_count == 4
        assert router.miss_count == 3
        assert router.miss_rate == 0.75

    def test_miss_rate_is_zero_before_any_routing(self):
        assert MetadataRouter(teacher_names=["a"]).miss_rate == 0.0

    def test_miss_rate_accumulates_across_batches(self):
        router = MetadataRouter(teacher_names=["a"], domain_to_index={"a": 0})
        for _ in range(3):
            router.route(
                {"teacher_id": ["a", "?"], "domain": ["a", "?"]},
                device=torch.device("cpu"),
            )
        assert router.routed_count == 6
        assert router.miss_rate == 0.5


class TestStrictMode:
    """Live teachers: an unroutable sample distils through the wrong expert."""

    def test_strict_raises_and_names_the_known_keys(self):
        router = MetadataRouter(
            teacher_names=["code_t"], domain_to_index={"code": 0}, strict=True
        )
        with pytest.raises(KeyError, match="teachers.teacher_ids"):
            router.route(
                {"teacher_id": ["nope"], "domain": ["nope"]},
                device=torch.device("cpu"),
            )

    def test_strict_allows_a_resolvable_batch(self):
        router = MetadataRouter(
            teacher_names=["code_t"], domain_to_index={"code": 0}, strict=True
        )
        idx = router.route(
            {"teacher_id": ["code_t"], "domain": ["code"]}, device=torch.device("cpu")
        )
        assert idx.tolist() == [0]

    def test_strict_accepts_the_domain_fallback(self):
        router = MetadataRouter(
            teacher_names=["code_t"], domain_to_index={"code": 0}, strict=True
        )
        idx = router.route(
            {"teacher_id": ["unknown-id"], "domain": ["code"]},
            device=torch.device("cpu"),
        )
        assert idx.tolist() == [0]
        assert router.miss_count == 0

    def test_default_is_tolerant(self):
        router = MetadataRouter(teacher_names=["code_t"])
        idx = router.route(
            {"teacher_id": ["nope"], "domain": ["nope"]}, device=torch.device("cpu")
        )
        assert idx.tolist() == [0]
        assert router.miss_count == 1


class TestCacheTeacherDomainNames:
    """Passing raw config entries used to poison both router lookups."""

    def test_non_string_domain_is_rejected(self):
        from hivemind.teacher_registry import create_hf_cache_teachers

        with pytest.raises(TypeError, match="domain_names"):
            create_hf_cache_teachers(
                domains=[{"name": "general", "layer_c": "org/repo"}],
                teacher_ids={"general": "general_t"},
                vocab_size=8,
                device=torch.device("cpu"),
            )

    def test_domain_names_extracts_plain_names(self):
        from hivemind.data.hf_loader import domain_names

        assert domain_names(
            [{"name": "general", "layer_c": "org/g"}, {"name": "code"}, "math"]
        ) == ["general", "code", "math"]

    def test_real_config_domains_route_end_to_end(self):
        """stream_small's shipped domains + teacher_ids must resolve cleanly."""
        from omegaconf import OmegaConf

        from hivemind.data.hf_loader import domain_names
        from hivemind.teacher_registry import create_hf_cache_teachers

        cfg = OmegaConf.load("configs/stream_small.yaml")
        names = domain_names(OmegaConf.to_container(cfg.data.domains, resolve=True))
        teachers = create_hf_cache_teachers(
            domains=names,
            teacher_ids=dict(cfg.teachers.teacher_ids),
            vocab_size=8,
            device=torch.device("cpu"),
        )
        router = MetadataRouter(
            teacher_names=[t.name for t in teachers],
            domain_to_index={t.model.domain: i for i, t in enumerate(teachers)},
        )
        # Route one sample per domain using the real Layer-C teacher_id values.
        meta = {
            "teacher_id": [dict(cfg.teachers.teacher_ids)[n] for n in names],
            "domain": list(names),
        }
        idx = router.route(meta, device=torch.device("cpu"))
        assert idx.tolist() == list(range(len(names)))
        assert router.miss_count == 0

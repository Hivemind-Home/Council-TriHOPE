"""Tests for MetadataRouter."""

from __future__ import annotations

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

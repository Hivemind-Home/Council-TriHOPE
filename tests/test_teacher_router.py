"""Tests for teacher router."""

import torch

from hivemind.teacher_router import RouterConfig, TeacherRouter


def test_route_picks_closest_teacher():
    """Router picks the teacher whose prototype is most similar."""
    torch.manual_seed(42)
    router = TeacherRouter(RouterConfig(hard_routing=True))

    # 2 teachers, embeddings clearly closer to teacher 1
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # [2, 2]
    embeddings = torch.tensor([[0.9, 0.1], [0.1, 0.9]])  # [2, 2]

    # Normalize
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    indices, scores = router.route(embeddings, prototypes)

    assert indices[0].item() == 0  # first sample closest to teacher 0
    assert indices[1].item() == 1  # second sample closest to teacher 1
    assert scores.shape == (2, 2)


def test_route_shape():
    """Router output shapes are correct."""
    torch.manual_seed(42)
    router = TeacherRouter()

    B, K, d = 4, 3, 16
    embeddings = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
    prototypes = torch.nn.functional.normalize(torch.randn(K, d), dim=-1)

    indices, scores = router.route(embeddings, prototypes)

    assert indices.shape == (B,)
    assert scores.shape == (B, K)
    assert indices.min() >= 0
    assert indices.max() < K

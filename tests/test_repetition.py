"""Tests for repetition signals."""

import torch

from hivemind.controller.repetition import (
    BucketSurpriseRepetition,
    FusedRepetition,
    MomentumRepetition,
    RetrievalHitRepetition,
)


def test_momentum_rises_with_aligned_gradients():
    """Momentum repetition rises when gradients consistently align with momentum."""
    rep = MomentumRepetition(alpha=0.3)

    # Simulate aligned gradients
    direction = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(20):
        score = rep.update(direction, direction * 2.0)  # aligned

    assert score > 0.5


def test_momentum_stays_low_with_orthogonal():
    """Momentum repetition stays low with orthogonal gradients."""
    rep = MomentumRepetition(alpha=0.3)

    grad = torch.tensor([1.0, 0.0])
    m = torch.tensor([0.0, 1.0])

    for _ in range(20):
        score = rep.update(grad, m)

    assert score < 0.2


def test_momentum_ignores_negative_cosine():
    """Negative cosine (conflict) doesn't contribute to repetition."""
    rep = MomentumRepetition(alpha=0.3)

    grad = torch.tensor([1.0, 0.0])
    m = torch.tensor([-1.0, 0.0])  # opposite

    for _ in range(20):
        score = rep.update(grad, m)

    assert score < 0.01


def test_bucket_surprise_repetition_rises():
    """Bucket repetition rises with repeated visits and decreasing surprise."""
    bucket_rep = BucketSurpriseRepetition(alpha=0.3, k=5.0, lookback_delta=5)

    # Visit bucket 0 many times with decreasing surprise
    scores = []
    for i in range(30):
        surprise = max(0.1, 5.0 - i * 0.2)
        s = bucket_rep.update(bucket_id=0, surprise=surprise)
        scores.append(s)

    # Later scores should be higher than early ones
    assert scores[-1] > scores[5]


def test_bucket_surprise_rare_bucket_low_score():
    """Rare buckets have low repetition score."""
    bucket_rep = BucketSurpriseRepetition(alpha=0.3, k=10.0, lookback_delta=5)

    # Single visit
    s = bucket_rep.update(bucket_id=99, surprise=1.0)
    # freq_confidence = 1/(1+10) ≈ 0.09
    assert s < 0.2


def test_retrieval_hit_rises_with_duplicates():
    """Retrieval hit rate rises when buffer contains near-duplicates."""
    rep = RetrievalHitRepetition(kappa=3.0, threshold=0.9, buffer_size=100)

    # Add same embedding multiple times
    emb = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(10):
        score = rep.update(emb)

    assert score > 0.5


def test_retrieval_hit_zero_for_novel():
    """Retrieval hit rate is 0 for first-ever embedding."""
    rep = RetrievalHitRepetition(kappa=3.0, threshold=0.9, buffer_size=100)
    emb = torch.randn(16)
    score = rep.update(emb)
    assert score == 0.0  # empty buffer, no hits


def test_fused_repetition():
    """Fused repetition combines all three signals."""
    from hivemind.controller.config import RepetitionConfig

    config = RepetitionConfig(
        lambda_mom=0.5, lambda_hash=0.5, lambda_ret=0.0,
        alpha_mom=0.3, alpha_hash=0.3,
    )
    fused = FusedRepetition(config)

    grad = torch.tensor([1.0, 0.0, 0.0])
    m = torch.tensor([1.0, 0.0, 0.0])

    score, components = fused.compute(
        grad=grad, adam_m_prev=m, bucket_id=0, surprise=1.0, embedding=None
    )

    assert 0.0 <= score <= 1.0
    assert "mom" in components
    assert "hash" in components

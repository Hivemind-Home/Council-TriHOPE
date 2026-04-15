"""Tests for shared routing embedding."""

import torch

from hivemind.embedding import SharedEmbedding


def test_embedding_shape(tiny_model, tiny_tokens):
    """SharedEmbedding produces [B, d] normalized vectors."""
    embed = SharedEmbedding(tiny_model.embed)
    h = embed(tiny_tokens)
    assert h.shape == (tiny_tokens.shape[0], tiny_model.config.dim)


def test_embedding_normalized(tiny_model, tiny_tokens):
    """Output vectors are L2-normalized."""
    embed = SharedEmbedding(tiny_model.embed)
    h = embed(tiny_tokens)
    norms = h.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embedding_deterministic(tiny_model, tiny_tokens):
    """Same input produces same output."""
    embed = SharedEmbedding(tiny_model.embed)
    h1 = embed(tiny_tokens)
    h2 = embed(tiny_tokens)
    assert torch.allclose(h1, h2)

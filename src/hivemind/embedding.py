"""Shared routing embedding for teacher selection.

Maps each input sample into a common routing space via the student's own
embedding layer + mean pooling. This avoids a separate encoder and creates
a natural routing space where teacher prototypes and samples coexist.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEmbedding(nn.Module):
    """Shared routing embedding: f_emb(x) → R^d.

    Reuses a token embedding layer (typically the student's) and applies
    mean pooling to produce a single routing vector per sample.
    """

    def __init__(self, embed_layer: nn.Embedding) -> None:
        super().__init__()
        self.embed = embed_layer
        self.dim = embed_layer.embedding_dim

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute routing embedding.

        Args:
            tokens: [B, T] token indices.

        Returns:
            h: [B, d] normalized routing vectors.
        """
        emb = self.embed(tokens)  # [B, T, d]
        h = emb.mean(dim=1)  # [B, d]
        # L2 normalize for clean cosine similarity downstream
        return F.normalize(h, p=2, dim=-1)

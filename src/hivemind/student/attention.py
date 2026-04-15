"""Causal self-attention with RoPE and LoRA adapters."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoRAConfig
from .lora import LoRALinear


def _precompute_rope_frequencies(
    dim: int, max_seq_len: int, base: float = 10000.0, device: torch.device | None = None
) -> torch.Tensor:
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    Args:
        x: [B, num_heads, T, head_dim]
        freqs: [T, head_dim//2] complex
    """
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, T, D)
    return x_rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and LoRA on Q, K, V, O projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        # Build projections — LoRA or plain Linear
        targets = set(lora_config.target_modules) if lora_config else set()

        def _make_proj(name: str) -> nn.Module:
            if name in targets and lora_config is not None:
                return LoRALinear(
                    dim, dim, rank=lora_config.rank,
                    alpha=lora_config.alpha, dropout=lora_config.dropout, bias=False,
                )
            return nn.Linear(dim, dim, bias=False)

        self.w_q = _make_proj("q")
        self.w_k = _make_proj("k")
        self.w_v = _make_proj("v")
        self.w_o = _make_proj("o")

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            _precompute_rope_frequencies(self.head_dim, max_seq_len, rope_base),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, dim]

        Returns:
            [B, T, dim]
        """
        B, T, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        q = self.w_q(x).view(B, T, H, HD).transpose(1, 2)  # [B, H, T, HD]
        k = self.w_k(x).view(B, T, H, HD).transpose(1, 2)
        v = self.w_v(x).view(B, T, H, HD).transpose(1, 2)

        # Apply RoPE to Q and K
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)

        # Scaled dot-product attention with causal mask (SDPA handles this)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, T, HD]

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(attn_out)

    def get_lora_modules(self) -> dict[str, LoRALinear]:
        """Return LoRA-enabled projections."""
        result = {}
        for name, proj in [("q", self.w_q), ("k", self.w_k), ("v", self.w_v), ("o", self.w_o)]:
            if isinstance(proj, LoRALinear):
                result[name] = proj
        return result

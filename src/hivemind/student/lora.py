"""LoRA (Low-Rank Adaptation) module implementation.

Implements LoRA adapters that decompose weight updates into low-rank matrices:
    ΔW = B @ A * (alpha / rank)
where A ∈ R^{r×d_in} and B ∈ R^{d_out×r}.

Supports merge/reset for F→P consolidation and Top-K masking for sparse updates.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with attached LoRA adapter.

    Output = base_linear(x) + (x @ A^T @ B^T) * (alpha / rank)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base (P-store) weight
        self.base = nn.Linear(in_features, out_features, bias=bias)

        # LoRA (F-store) parameters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora()

    def reset_lora(self) -> None:
        """Reset LoRA parameters: A with Kaiming, B with zeros (standard LoRA init)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: base output + LoRA delta."""
        base_out = self.base(x)
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge_lora_into_base(self) -> None:
        """Consolidate F→P: W_base += B @ A * scaling, then reset LoRA."""
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.add_(delta)
        self.reset_lora()

    @property
    def base_params(self) -> list[nn.Parameter]:
        """P-store parameters (base weight + optional bias)."""
        params = [self.base.weight]
        if self.base.bias is not None:
            params.append(self.base.bias)
        return params

    @property
    def lora_params(self) -> list[nn.Parameter]:
        """F-store parameters (LoRA A and B matrices)."""
        return [self.lora_A, self.lora_B]

    def get_top_k_mask(
        self,
        k: int,
        granularity: Literal["rank_components", "rows"] = "rank_components",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Top-K masks for sparse LoRA updates.

        Args:
            k: Number of top elements to keep.
            granularity: "rank_components" scores each rank direction,
                        "rows" scores each output row of the delta.

        Returns:
            (mask_A, mask_B): Boolean masks for A and B parameters.
        """
        if self.lora_A.grad is None or self.lora_B.grad is None:
            return torch.ones_like(self.lora_A, dtype=torch.bool), torch.ones_like(
                self.lora_B, dtype=torch.bool
            )

        if granularity == "rank_components":
            # Score each rank component by gradient energy in both A and B
            # s_k = ||∇B[:,k]||₂ + ||∇A[k,:]||₂
            scores = self.lora_B.grad.norm(dim=0) + self.lora_A.grad.norm(dim=1)  # [rank]
            k_clamped = min(k, self.rank)
            _, top_indices = scores.topk(k_clamped)
            mask_A = torch.zeros(self.rank, dtype=torch.bool, device=self.lora_A.device)
            mask_B = torch.zeros(self.rank, dtype=torch.bool, device=self.lora_B.device)
            mask_A[top_indices] = True
            mask_B[top_indices] = True
            # Expand to full parameter shape
            mask_A = mask_A.unsqueeze(1).expand_as(self.lora_A)
            mask_B = mask_B.unsqueeze(0).expand_as(self.lora_B)
            return mask_A, mask_B

        elif granularity == "rows":
            # Score each row of ΔW = B @ A by gradient norm
            # We use B's row gradient norms as proxy
            row_scores = self.lora_B.grad.norm(dim=1)  # [out_features]
            k_clamped = min(k, self.out_features)
            _, top_indices = row_scores.topk(k_clamped)
            mask_B = torch.zeros(
                self.out_features, dtype=torch.bool, device=self.lora_B.device
            )
            mask_B[top_indices] = True
            mask_B = mask_B.unsqueeze(1).expand_as(self.lora_B)
            mask_A = torch.ones_like(self.lora_A, dtype=torch.bool)
            return mask_A, mask_B

        else:
            raise ValueError(f"Unknown granularity: {granularity}")

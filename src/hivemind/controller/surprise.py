"""Adam-based surprise signal.

S_t(j) = (1/d_j) · Σ_i g²_{t,i}^(j) / (v_{t-1,i}^(j) + ε)

Measures how much the current gradient exceeds Adam's expected scale
for each parameter coordinate, averaged over the module.
"""

from __future__ import annotations

import torch

from .config import SurpriseConfig


class AdamSurprise:
    """Computes Adam-based surprise for a module."""

    def __init__(self, config: SurpriseConfig | None = None) -> None:
        self.config = config or SurpriseConfig()

    def compute(
        self,
        grad: torch.Tensor,
        adam_v_prev: torch.Tensor,
    ) -> float:
        """Compute Adam-based surprise score.

        S_t(j) = (1/d_j) · Σ_i g²_{t,i} / (v_{t-1,i} + ε)

        Uses v_{t-1} (causal: BEFORE current step updates Adam).

        Args:
            grad: flattened gradient vector g_t^(j).
            adam_v_prev: flattened Adam second moment v_{t-1}^(j).

        Returns:
            Clamped surprise scalar.
        """
        eps = self.config.eps
        d_j = grad.numel()

        if d_j == 0:
            return 0.0

        # Elementwise: g²_i / (v_{t-1,i} + ε)
        surprise = (grad.float().pow(2) / (adam_v_prev.float() + eps)).sum().item() / d_j

        # Clamp to prevent pathological batches from hijacking controller
        return max(self.config.s_min, min(self.config.s_max, surprise))

    def compute_from_param_list(
        self,
        params: list[torch.nn.Parameter],
        adam_v_list: list[torch.Tensor],
    ) -> float:
        """Compute surprise from lists of parameters and their Adam v states.

        Concatenates all param gradients and v states, then computes surprise.
        """
        grads = []
        vs = []
        for p, v in zip(params, adam_v_list):
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
                vs.append(v.detach().flatten())

        if not grads:
            return 0.0

        grad_cat = torch.cat(grads)
        v_cat = torch.cat(vs)
        return self.compute(grad_cat, v_cat)

"""Stability signals for the R/F/P controller.

Three complementary signals:
1. Directional consistency: C_t(j) = cos(g_t(j), m_{t-1}(j))
2. Adam ratio stability: Stab_t(j) = (1/d_j) · Σ m²_{t-1,i} / (v_{t-1,i} + ε)
3. Windowed gradient variance: V_t(j) from EMAs of norm and squared norm
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import StabilityConfig


class StabilityTracker:
    """Tracks stability signals per module over time."""

    def __init__(self, config: StabilityConfig | None = None) -> None:
        self.config = config or StabilityConfig()
        # Windowed variance EMA states
        self._ema_norm: float = 0.0  # a_t: EMA of gradient norm
        self._ema_norm_sq: float = 0.0  # b_t: EMA of squared gradient norm
        self._initialized: bool = False

    def compute_directional(
        self,
        grad: torch.Tensor,
        adam_m_prev: torch.Tensor,
    ) -> float:
        """Directional stability: C_t(j) = cos(g_t(j), m_{t-1}(j)).

        Args:
            grad: flattened gradient g_t^(j).
            adam_m_prev: flattened Adam first moment m_{t-1}^(j).

        Returns:
            Cosine similarity in [-1, 1], or 0.0 if m_{t-1} is too small.
        """
        m_norm = adam_m_prev.float().norm()
        if m_norm < self.config.warmup_threshold:
            return 0.0

        g_flat = grad.float().flatten()
        m_flat = adam_m_prev.float().flatten()

        cos_sim = F.cosine_similarity(g_flat.unsqueeze(0), m_flat.unsqueeze(0)).item()
        return cos_sim

    def compute_adam_ratio(
        self,
        adam_m_prev: torch.Tensor,
        adam_v_prev: torch.Tensor,
    ) -> float:
        """Adam ratio stability: Stab_t(j) = (1/d_j) · Σ m²_{t-1,i} / (v_{t-1,i} + ε).

        Measures signal-to-noise: how much of the gradient energy is
        consistent direction vs random fluctuation.

        Args:
            adam_m_prev: flattened Adam first moment m_{t-1}^(j).
            adam_v_prev: flattened Adam second moment v_{t-1}^(j).

        Returns:
            Stability score in [0, 1].
        """
        eps = self.config.eps
        m = adam_m_prev.float().flatten()
        v = adam_v_prev.float().flatten()
        d_j = m.numel()

        if d_j == 0:
            return 0.0

        ratio = (m.pow(2) / (v + eps)).sum().item() / d_j
        return max(0.0, min(1.0, ratio))

    def update_windowed_variance(self, grad_norm: float) -> float:
        """Windowed gradient variance: V_t(j).

        Tracks EMAs of gradient norm and squared norm, then computes:
            V_t = (b_t - a_t²) / (a_t² + ε)

        Args:
            grad_norm: r_t(j) = ||g_t(j)||₂

        Returns:
            Normalized windowed variance (higher = more volatile).
        """
        alpha = self.config.ema_alpha
        eps = self.config.eps

        if not self._initialized:
            self._ema_norm = grad_norm
            self._ema_norm_sq = grad_norm**2
            self._initialized = True
            return 0.0

        # Update EMAs
        self._ema_norm = (1 - alpha) * self._ema_norm + alpha * grad_norm
        self._ema_norm_sq = (1 - alpha) * self._ema_norm_sq + alpha * (grad_norm**2)

        # Variance proxy: E[r²] - E[r]²
        variance = self._ema_norm_sq - self._ema_norm**2
        variance = max(0.0, variance)  # numerical safety

        # Normalize by mean²
        denominator = self._ema_norm**2 + eps
        return variance / denominator

    def compute_directional_from_params(
        self,
        params: list[torch.nn.Parameter],
        adam_m_list: list[torch.Tensor],
    ) -> float:
        """Compute directional stability from parameter and moment lists."""
        grads = []
        ms = []
        for p, m in zip(params, adam_m_list):
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
                ms.append(m.detach().flatten())
        if not grads:
            return 0.0
        return self.compute_directional(torch.cat(grads), torch.cat(ms))

    def compute_adam_ratio_from_params(
        self,
        adam_m_list: list[torch.Tensor],
        adam_v_list: list[torch.Tensor],
    ) -> float:
        """Compute Adam ratio from moment lists."""
        if not adam_m_list:
            return 0.0
        m_cat = torch.cat([m.detach().flatten() for m in adam_m_list])
        v_cat = torch.cat([v.detach().flatten() for v in adam_v_list])
        return self.compute_adam_ratio(m_cat, v_cat)

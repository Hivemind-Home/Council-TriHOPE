"""Repetition signals for the R/F/P controller.

Three complementary signals:
1. Momentum repetition: R_t(mom) = EMA of max(0, cos(g_t, m_{t-1}))
2. Bucket surprise decay: R_t(hash) = freq_confidence · σ(surprise_trend)
3. Retrieval hit rate: R_t(ret) = 1 - exp(-H_t/κ)
Fused: R_t = λ₁·R_mom + λ₂·R_hash + λ₃·R_ret
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from .config import RepetitionConfig


class MomentumRepetition:
    """Repetition via persistent directional agreement (optimizer-only).

    R_t(mom,j) = (1 - α_R) · R_{t-1} + α_R · max(0, cos(g_t(j), m_{t-1}(j)))
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._score: float = 0.0

    def update(self, grad: torch.Tensor, adam_m_prev: torch.Tensor) -> float:
        """Update and return momentum repetition score."""
        g = grad.float().flatten()
        m = adam_m_prev.float().flatten()

        m_norm = m.norm()
        if m_norm < 1e-8:
            cos_val = 0.0
        else:
            cos_val = F.cosine_similarity(g.unsqueeze(0), m.unsqueeze(0)).item()

        # Only positive alignment contributes to repetition
        positive_cos = max(0.0, cos_val)
        self._score = (1 - self.alpha) * self._score + self.alpha * positive_cos
        return self._score

    @property
    def score(self) -> float:
        return self._score


class BucketSurpriseRepetition:
    """Repetition via surprise decay across similar samples.

    R_t(hash) = [n_t(h) / (n_t(h) + k)] · σ(S̄_{t-Δ}(h) - S̄_t(h))

    Uses event-time EMA: only update bucket stats when that bucket is visited.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        k: float = 10.0,
        lookback_delta: int = 100,
    ) -> None:
        self.alpha = alpha
        self.k = k
        self.lookback_delta = lookback_delta

        # Per-bucket state
        self._surprise_ema: dict[int, float] = defaultdict(float)
        self._surprise_history: dict[int, list[float]] = defaultdict(list)
        self._count: dict[int, int] = defaultdict(int)

    def update(self, bucket_id: int, surprise: float) -> float:
        """Update bucket state and compute repetition score.

        Args:
            bucket_id: hash/cluster ID of the current sample.
            surprise: current surprise score S_t.

        Returns:
            R_t(hash) score in [0, 1].
        """
        # Update bucket frequency
        self._count[bucket_id] += 1
        n = self._count[bucket_id]

        # Update bucket surprise EMA (event-time: only when visited)
        old_ema = self._surprise_ema[bucket_id]
        new_ema = (1 - self.alpha) * old_ema + self.alpha * surprise
        self._surprise_ema[bucket_id] = new_ema

        # Track history for trend computation
        self._surprise_history[bucket_id].append(new_ema)

        # Frequency confidence: n / (n + k)
        freq_confidence = n / (n + self.k)

        # Surprise trend: σ(S̄_{t-Δ} - S̄_t)
        history = self._surprise_history[bucket_id]
        if len(history) > self.lookback_delta:
            past_ema = history[-self.lookback_delta - 1]
            trend_input = past_ema - new_ema  # positive = surprise decreasing
        else:
            trend_input = 0.0

        trend = _sigmoid(trend_input)

        return freq_confidence * trend

    def get_score(self, bucket_id: int) -> float:
        """Get the current repetition score for a bucket without updating."""
        n = self._count.get(bucket_id, 0)
        if n == 0:
            return 0.0
        freq_confidence = n / (n + self.k)
        history = self._surprise_history.get(bucket_id, [])
        if len(history) > self.lookback_delta:
            past_ema = history[-self.lookback_delta - 1]
            current_ema = history[-1]
            trend_input = past_ema - current_ema
        else:
            trend_input = 0.0
        return freq_confidence * _sigmoid(trend_input)


class RetrievalHitRepetition:
    """Repetition via retrieval hit rate (embedding similarity).

    R_t(ret) = 1 - exp(-H_t / κ)
    where H_t = count of cos(e_t, e) ≥ τ for e in buffer.
    """

    def __init__(
        self,
        kappa: float = 5.0,
        threshold: float = 0.85,
        buffer_size: int = 1000,
    ) -> None:
        self.kappa = kappa
        self.threshold = threshold
        self.buffer_size = buffer_size
        self._buffer: list[torch.Tensor] = []

    def update(self, embedding: torch.Tensor) -> float:
        """Update buffer and compute retrieval hit rate.

        Args:
            embedding: [d] embedding vector of current sample.

        Returns:
            R_t(ret) score in [0, 1].
        """
        e = embedding.detach().float().flatten()

        # Count near-neighbors in buffer
        hit_count = 0
        if self._buffer:
            buffer_tensor = torch.stack(self._buffer)  # [N, d]
            sims = F.cosine_similarity(e.unsqueeze(0), buffer_tensor)  # [N]
            hit_count = (sims >= self.threshold).sum().item()

        # Add to buffer (FIFO)
        self._buffer.append(e)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

        # R_t(ret) = 1 - exp(-H_t / κ)
        return 1.0 - math.exp(-hit_count / max(self.kappa, 1e-8))


class FusedRepetition:
    """Fused repetition score: R_t = λ₁·R_mom + λ₂·R_hash + λ₃·R_ret."""

    def __init__(self, config: RepetitionConfig | None = None) -> None:
        self.config = config or RepetitionConfig()
        self.momentum = MomentumRepetition(alpha=self.config.alpha_mom)
        self.bucket = BucketSurpriseRepetition(
            alpha=self.config.alpha_hash,
            k=self.config.bucket_smoothing_k,
            lookback_delta=self.config.lookback_delta,
        )
        self.retrieval = RetrievalHitRepetition(
            kappa=self.config.kappa,
            threshold=self.config.sim_threshold,
            buffer_size=self.config.buffer_size,
        )

    def compute(
        self,
        grad: torch.Tensor,
        adam_m_prev: torch.Tensor,
        bucket_id: int,
        surprise: float,
        embedding: torch.Tensor | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute fused repetition score.

        Returns:
            (fused_score, {"mom": ..., "hash": ..., "ret": ...})
        """
        r_mom = self.momentum.update(grad, adam_m_prev)
        r_hash = self.bucket.update(bucket_id, surprise)

        r_ret = 0.0
        if embedding is not None and self.config.lambda_ret > 0:
            r_ret = self.retrieval.update(embedding)

        # Normalize weights
        lam1 = self.config.lambda_mom
        lam2 = self.config.lambda_hash
        lam3 = self.config.lambda_ret if embedding is not None else 0.0
        total_lam = lam1 + lam2 + lam3
        if total_lam > 0:
            lam1 /= total_lam
            lam2 /= total_lam
            lam3 /= total_lam

        fused = lam1 * r_mom + lam2 * r_hash + lam3 * r_ret

        return fused, {"mom": r_mom, "hash": r_hash, "ret": r_ret}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

"""Controller configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SurpriseConfig:
    """Adam-based surprise signal configuration."""

    s_min: float = 0.0
    s_max: float = 20.0
    eps: float = 1e-8


@dataclass
class StabilityConfig:
    """Stability signal configuration."""

    ema_alpha: float = 0.1
    warmup_threshold: float = 1e-6
    eps: float = 1e-8


@dataclass
class RepetitionConfig:
    """Repetition signal configuration."""

    alpha_mom: float = 0.05
    alpha_hash: float = 0.1
    kappa: float = 5.0
    sim_threshold: float = 0.85
    bucket_smoothing_k: float = 10.0
    lookback_delta: int = 100
    lambda_mom: float = 0.3
    lambda_hash: float = 0.5
    lambda_ret: float = 0.2
    buffer_size: int = 1000


@dataclass
class PolicyConfig:
    """R/F/P routing policy configuration."""

    top_m_modules: int = 6
    surprise_high: float = 2.0
    repetition_low: float = 0.3
    repetition_medium: float = 0.5
    stability_high_C: float = 0.5
    stability_low_V: float = 0.3


@dataclass
class ConsolidationConfig:
    """F→P consolidation configuration."""

    period: int = 1000
    min_stability_C: float = 0.7
    min_repetition: float = 0.6
    merge_strategy: str = "direct"


@dataclass
class WriterConfig:
    """Write executor configuration."""

    top_k_granularity: str = "rank_components"
    top_k_fraction: float = 0.5


@dataclass
class ControllerConfig:
    """Full controller configuration."""

    surprise: SurpriseConfig = field(default_factory=SurpriseConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)

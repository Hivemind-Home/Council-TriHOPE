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
    """F→P consolidation configuration.

    merge_strategy:
    - ``direct``: merge LoRA deltas into base weight and reset LoRA
      (fast, Theory 101 §8 option 2). Always safe.
    - ``distill``: capture the student+LoRA behaviour on a replay batch,
      train the base weights of the target block to match it, then
      reset LoRA (Theory 101 §8 option 1). Requires a recent-batch
      replay buffer; tunable via ``distill_*`` fields.
    """

    period: int = 1000
    min_stability_C: float = 0.7
    min_repetition: float = 0.6
    merge_strategy: str = "direct"
    distill_iters: int = 4
    distill_lr: float = 1.0e-4
    distill_tau: float = 2.0
    distill_replay_size: int = 16
    # Save a tagged rollback checkpoint immediately before any merge
    # (needed by the incorrect-consolidation / rollback experiments).
    checkpoint_before_merge: bool = False


@dataclass
class WriterConfig:
    """Write executor configuration."""

    top_k_granularity: str = "rank_components"
    top_k_fraction: float = 0.5


#: Canonical signal names accepted in ``AblationConfig.disable_signals``,
#: mapping aliases (paper terminology) onto internal names.
SIGNAL_ALIASES = {
    "surprise": "surprise",
    "repetition": "repetition",
    "repetition_mom": "repetition_mom",
    "repetition_hash": "repetition_hash",
    "repetition_ret": "repetition_ret",
    "stability_c": "stability_C",
    "cosine": "stability_C",
    "cosine_alignment": "stability_C",
    "stability_v": "stability_V",
    "volatility": "stability_V",
    "teacher_confidence": "teacher_confidence",
}


@dataclass
class AblationConfig:
    """Per-signal ablation switches for the routing-signal study.

    A disabled signal is *not computed* (its tracker state does not update)
    and is pinned to a neutral value chosen so the policy degrades to the
    sub-policy over the remaining signals:

    - ``surprise`` disabled → pinned to ``policy.surprise_high``: the
      R-branch ``S ≥ θ_S ∧ R < θ_R`` reduces to the pure recurrence test.
    - ``repetition`` disabled → pinned to ``policy.repetition_medium``: the
      P-branch's recurrence conjunct passes (stability alone decides P)
      while the R-branch's ``R < θ_R`` fails — a controller that cannot
      measure recurrence never claims "not recurring".
    - ``stability_C`` / ``stability_V`` disabled → pinned to their passing
      thresholds (P is decided by the remaining stability evidence).
    - ``repetition_mom|hash|ret`` disabled → that component's λ-weight is
      zeroed and the remaining λs renormalize.
    - ``teacher_confidence`` disabled → forces ``use_teacher_confidence``
      off regardless of its value.

    ``disable_stores`` removes whole branches: ``R`` (no-retrieval
    baseline — R-routed batches fall through to F) and ``P``
    (no-consolidation baseline — P branch and the consolidation sweep are
    both disabled).
    """

    disable_signals: list[str] = field(default_factory=list)
    disable_stores: list[str] = field(default_factory=list)
    top_m_override: int | None = None
    # Neutral overrides; ``None`` resolves against the policy thresholds
    # as documented above.
    neutral_surprise: float | None = None
    neutral_repetition: float | None = None
    neutral_stability_C: float | None = None
    neutral_stability_V: float | None = None
    # Teacher-confidence gate on the policy (separate from the loss-level
    # confidence weighting in DistillationConfig).
    use_teacher_confidence: bool = False
    confidence_threshold: float = 0.5
    confidence_gate: str = "no_p"  # "no_p" (demote P→F) | "force_r"

    def __post_init__(self) -> None:
        canonical: list[str] = []
        for name in self.disable_signals:
            key = str(name).lower()
            if key not in SIGNAL_ALIASES:
                raise ValueError(
                    f"Unknown signal '{name}' in ablation.disable_signals; "
                    f"allowed: {sorted(set(SIGNAL_ALIASES))}"
                )
            canonical.append(SIGNAL_ALIASES[key])
        self.disable_signals = canonical
        for store in self.disable_stores:
            if store not in ("R", "P"):
                raise ValueError(
                    f"Unknown store '{store}' in ablation.disable_stores; "
                    "allowed: ['R', 'P']"
                )
        if self.confidence_gate not in ("no_p", "force_r"):
            raise ValueError(
                f"Unknown confidence_gate '{self.confidence_gate}'; "
                "allowed: 'no_p' | 'force_r'"
            )
        if "teacher_confidence" in self.disable_signals:
            self.use_teacher_confidence = False

    def is_disabled(self, signal: str) -> bool:
        return signal in self.disable_signals


@dataclass
class DebugConfig:
    """Experiment-harness debug controls (incorrect-consolidation study).

    ``force_consolidate_steps`` merges ALL F-modules at the given steps,
    bypassing threshold re-validation — the "consolidate at the wrong time"
    probe. ``policy_override`` short-circuits classification entirely.
    Combine with ``consolidation.checkpoint_before_merge`` for rollback.
    """

    force_consolidate_steps: list[int] = field(default_factory=list)
    policy_override: str | None = None  # None | always_p | always_f | always_r

    def __post_init__(self) -> None:
        if self.policy_override not in (None, "always_p", "always_f", "always_r"):
            raise ValueError(
                f"Unknown policy_override '{self.policy_override}'; "
                "allowed: always_p | always_f | always_r"
            )
        self.force_consolidate_steps = [int(s) for s in self.force_consolidate_steps]


@dataclass
class ControllerConfig:
    """Full controller configuration."""

    enabled: bool = True
    surprise: SurpriseConfig = field(default_factory=SurpriseConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

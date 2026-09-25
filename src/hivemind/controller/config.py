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
    # EMA smoothing for the *sustained* directional cosine (C̄). The
    # instantaneous C = cos(g_t, m_{t-1}) is noisy at any single step; C̄
    # accumulates it so a module that has been directionally consistent for
    # a sustained window reads high even when the current step is noisy.
    c_ema_alpha: float = 0.1
    warmup_threshold: float = 1e-6
    eps: float = 1e-8


@dataclass
class MomentsConfig:
    """Where the controller's ``m_{t-1}`` / ``v_{t-1}`` evidence comes from.

    ``tracked`` (default): the controller keeps its own bias-corrected
    Adam-style moments for every indexed coordinate, advanced from the full
    gradient every step regardless of the write mask (MoLF's universal
    momentum tracking, confined to the signal path; see
    ``controller/moments.py``). Identical to Adam's state on coordinates
    that are open every step; alive on coordinates the mask keeps closed.

    ``optimizer``: read MaskedAdamW's own ``exp_avg`` / ``exp_avg_sq``. Under
    exact masking these are zero for every never-opened coordinate, so
    surprise saturates and the cosine is a warmup zero on base-weight
    modules until P has fired — which needs those signals. Kept as the
    ablation that shows why ``tracked`` is necessary.

    ``sketch_stride``: track every k-th coordinate only (memory ÷ k; the
    signals are per-coordinate means and cosines, so a fixed stride is an
    unbiased, deterministic estimate). ``bias_correct``: divide by
    ``1 - β^t`` as Adam does before forming ``g²/v`` and ``m²/v``.
    """

    source: str = "tracked"
    sketch_stride: int = 1
    bias_correct: bool = True

    def __post_init__(self) -> None:
        if self.source not in ("tracked", "optimizer"):
            raise ValueError(
                f"moments.source must be 'tracked' or 'optimizer', got {self.source!r}"
            )
        if int(self.sketch_stride) < 1:
            raise ValueError(f"moments.sketch_stride must be >= 1, got {self.sketch_stride}")


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
    # Which directional-stability signal the P branch compares against
    # ``stability_high_C``:
    # - ``instant``   : C = cos(g_t, m_{t-1}) at this step. Noise at any single
    #   step on real text, so P effectively never fires (the original
    #   behaviour and the default).
    # - ``sustained`` : the EMA C̄ of that cosine (StabilityConfig.c_ema_alpha).
    #   Paper motivation: a module is promoted when it has been directionally
    #   consistent over a sustained window, not when one batch happens to be
    #   aligned — and it is the answer to the "cos(g, m) is noisy" critique.
    stability_source: str = "instant"
    # Controller family (task T3 — the E1 baselines share the stream, the
    # student and the write machinery; only the decision rule differs):
    # - ``rfp``           : TriHOPE's three-way R/F/P rule (default).
    # - ``surprise_only`` : Titans-style gate on surprise alone — R when
    #   S ≥ surprise_high (F if R is disabled), otherwise F; never P.
    # - ``adam_score``    : MoLF-style two-tier routing on the Adam SNR
    #   m²/v (``stability_adam``): P when ≥ adam_score_high, else F; never R.
    #   Pair with ``WriterConfig.flag_p_for_consolidation=false`` and
    #   ``consolidation.period=0`` — MoLF has no merge.
    # - ``teacher_partition`` : Gradient-Routing-style (Cloud et al. 2024).
    #   The LoRA rank is split into one contiguous slice per registered
    #   teacher; every F-type module opens exactly the batch teacher's slice
    #   (base closed, no consolidation). Uses the teacher LABEL — which
    #   TriHOPE never reads — and is the E5 baseline whose "unlearning" is
    #   ablating a slice (``DebugConfig.ablate_teacher_slice``).
    mode: str = "rfp"
    adam_score_high: float = 0.5
    # How ``adam_score`` decides (MoLF, arXiv:2605.07111):
    # - ``epd_argmax`` (default; faithful): per block, the dense expert (the
    #   block's base weights) and the LoRA expert compete on MoLF's Expected
    #   Preconditioned Descent score S_i = (lr_i / N_i) · Σ m² / (√v + ε)
    #   (their Eq. 4) and the winner alone updates — P opens the base
    #   weights, F opens the whole adapter. Every block routes every step.
    # - ``snr_threshold``: P iff mean(m²/(v+ε)) ≥ adam_score_high, else F —
    #   up to ε placement the square of MoLF's PFN ablation baseline.
    adam_score_rule: str = "epd_argmax"
    epd_lr_base: float = 1.0   # lr_i in the EPD score (the optimizer's lr_base / lr_lora)
    epd_lr_lora: float = 1.0

    def __post_init__(self) -> None:
        if self.stability_source not in ("instant", "sustained"):
            raise ValueError(
                f"Unknown policy.stability_source '{self.stability_source}'; "
                "allowed: 'instant' | 'sustained'"
            )
        if self.mode not in POLICY_MODES:
            raise ValueError(
                f"Unknown policy.mode '{self.mode}'; allowed: {sorted(POLICY_MODES)}"
            )
        if self.adam_score_rule not in ("epd_argmax", "snr_threshold"):
            raise ValueError(
                f"Unknown policy.adam_score_rule '{self.adam_score_rule}'; "
                "allowed: 'epd_argmax' | 'snr_threshold'"
            )


#: Controller families selectable through ``PolicyConfig.mode``.
POLICY_MODES = ("rfp", "surprise_only", "adam_score", "teacher_partition")


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
    # What decides that adapters are merged into base weights:
    # - ``signals`` : the periodic sweep re-validates flagged modules on
    #   stability ∧ recurrence (TriHOPE; default).
    # - ``plateau`` : Online-LoRA-style — merge ALL adapters whenever the
    #   training loss has plateaued (relative improvement over
    #   ``plateau_window`` steps below ``plateau_tolerance``), at most once
    #   per ``plateau_min_gap`` steps. The signal sweep is disabled. Works
    #   with ``controller.enabled=false`` (the E1 `plateau_trigger` baseline).
    trigger: str = "signals"
    plateau_window: int = 200
    plateau_tolerance: float = 0.01
    plateau_min_gap: int = 200
    # Online-LoRA (Wei et al., WACV 2025, §3.2) only consolidates on a
    # plateau that FOLLOWS a loss peak — the peak marks the distribution
    # shift, the plateau marks "the new distribution has been learned".
    # With ``true`` a plateau only fires after the window mean has risen by
    # more than the window's standard deviation since the last fire.
    plateau_require_peak: bool = False
    # Which directional-stability signal the merge re-validation reads:
    # - ``instant``   : the per-step cosine C = cos(g_t, m_{t-1}). Sampled
    #   at the exact period boundary, so a genuinely-stable module still
    #   fails unless the boundary batch happens to hit it aligned. This is
    #   the original behaviour and the default (nothing changes unless a
    #   config opts in).
    # - ``sustained`` : the EMA C̄ of that cosine (StabilityConfig.c_ema_alpha).
    #   Faithful to Theory 101 §"stability has remained high for multiple
    #   windows" — a module consolidates when it has been consistently
    #   aligned, not when it is lucky at one step. Keep ``min_stability_C``
    #   at its strict value; this changes *what is measured*, not the bar.
    stability_mode: str = "instant"
    merge_strategy: str = "direct"
    distill_iters: int = 4
    distill_lr: float = 1.0e-4
    distill_tau: float = 2.0
    distill_replay_size: int = 16
    # Save a tagged rollback checkpoint immediately before any merge
    # (needed by the incorrect-consolidation / rollback experiments).
    checkpoint_before_merge: bool = False

    def __post_init__(self) -> None:
        if self.stability_mode not in ("instant", "sustained"):
            raise ValueError(
                f"Unknown consolidation.stability_mode '{self.stability_mode}'; "
                "allowed: 'instant' | 'sustained'"
            )
        if self.trigger not in ("signals", "plateau"):
            raise ValueError(
                f"Unknown consolidation.trigger '{self.trigger}'; "
                "allowed: 'signals' | 'plateau'"
            )
        if int(self.plateau_window) < 2:
            raise ValueError("consolidation.plateau_window must be >= 2")
        self.plateau_window = int(self.plateau_window)
        self.plateau_min_gap = int(self.plateau_min_gap)


@dataclass
class WriterConfig:
    """Write executor configuration."""

    top_k_granularity: str = "rank_components"
    top_k_fraction: float = 0.5
    # A P action on an F-type module opens the whole adapter for one step
    # and, by default, flags the block so the next consolidation sweep can
    # merge it. ``false`` keeps the open-adapter step but never flags — the
    # MoLF-style baseline, which routes between adapter and base but has no
    # merge path.
    flag_p_for_consolidation: bool = True


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

    ``policy_override="random_matched"`` is the E1 control that proves the
    signals carry information: every selected module draws its store label
    from a fixed per-phase categorical distribution — TriHOPE's own action
    shares, read from ``random_shares_path`` (an ``action_share_by_phase.csv``
    written by ``analysis.run_report`` on a completed run, filtered to
    ``random_shares_spec_id``) — so the write budget is matched while the
    assignment is shuffled. Draws come from a generator seeded by
    ``(train.seed, step[, module rank])`` so every rank draws the same
    label and the global RNG is untouched. ``random_unit`` picks whether
    each module draws independently (``module``) or the whole step shares
    one draw (``step``). A missing file falls back to uniform thirds with
    a warning.
    """

    force_consolidate_steps: list[int] = field(default_factory=list)
    policy_override: str | None = None  # None | always_p | always_f | always_r | random_matched | random_commit | quota_shuffle
    # Selective rollback (task T5): a P action is demoted to F while the
    # batch's teacher is listed, and on resume every module whose pending
    # attribution is dominated (share ≥ block_min_share) by a listed teacher
    # has its adapter reset and its consolidation flag dropped — the
    # blocked teacher's tentative evidence is discarded instead of being
    # merged by a later, clean P action.
    block_p_for_teachers: list[str] = field(default_factory=list)
    block_min_share: float = 0.5
    # Gradient-routing ablation (T6): ``{step: int, teacher: str}`` zeroes
    # that teacher's rank slice (rows of A, columns of B) in every adapter
    # right after the optimizer step at ``step``, together with the slice's
    # optimizer state. A zeroed slice is dead thereafter (its gradients
    # vanish), which is exactly Gradient Routing's "remove the region".
    ablate_teacher_slice: dict | None = None
    random_shares_path: str | None = None
    random_shares_spec_id: str = "trihope"
    random_unit: str = "module"  # module | step

    def __post_init__(self) -> None:
        if self.policy_override not in (
            None, "always_p", "always_f", "always_r", "random_matched", "random_commit",
            "quota_shuffle"
        ):
            raise ValueError(
                f"Unknown policy_override '{self.policy_override}'; "
                "allowed: always_p | always_f | always_r | random_matched | random_commit | quota_shuffle"
            )
        if self.random_unit not in ("module", "step"):
            raise ValueError(
                f"Unknown random_unit '{self.random_unit}'; allowed: module | step"
            )
        self.force_consolidate_steps = [int(s) for s in self.force_consolidate_steps]
        self.block_p_for_teachers = [str(t) for t in self.block_p_for_teachers]
        if self.ablate_teacher_slice is not None:
            raw = dict(self.ablate_teacher_slice)
            if "step" not in raw or "teacher" not in raw:
                raise ValueError(
                    "ablate_teacher_slice needs {step: int, teacher: str}"
                )
            self.ablate_teacher_slice = {"step": int(raw["step"]), "teacher": str(raw["teacher"])}
        if not 0.0 <= float(self.block_min_share) <= 1.0:
            raise ValueError("block_min_share must be in [0, 1]")
        self.block_min_share = float(self.block_min_share)


@dataclass
class RetrievalConfig:
    """R-store replay: turn a deferred observation back into a fast write.

    Paper motivation (reframe §3.1 / task T2): an R action must be a
    *deferral*, not a dropped update. With ``replay_on_hit`` the R-store
    keeps the representative row of every R-routed batch; when the same
    bucket recurs and the controller stops routing it to R (the pattern has
    started repeating), the parked rows are replayed into the fast store as
    an extra F micro-step inside the same training step. Default off →
    the R tier is write-only, exactly as before (the ``r_terminal``
    ablation).
    """

    replay_on_hit: bool = False
    # A bucket is replayed once it holds R entries from at least this many
    # DISTINCT earlier steps and the current step routed it somewhere other
    # than R. (The writer appends one entry per R action, so entries are
    # grouped by their origin step.)
    hit_threshold: int = 2
    # Parked rows replayed per trigger, oldest first (FIFO).
    replay_batches: int = 1
    # Store the replayed update is routed to. Only "F" is implemented: the
    # replay opens the Top-K LoRA components of the Top-M F-type modules.
    replay_target: str = "F"

    def __post_init__(self) -> None:
        if self.replay_target != "F":
            raise ValueError(
                f"retrieval.replay_target={self.replay_target!r} is not supported; "
                "only 'F' is implemented"
            )
        if int(self.hit_threshold) < 1:
            raise ValueError("retrieval.hit_threshold must be >= 1")
        if int(self.replay_batches) < 1:
            raise ValueError("retrieval.replay_batches must be >= 1")
        self.hit_threshold = int(self.hit_threshold)
        self.replay_batches = int(self.replay_batches)


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
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    moments: MomentsConfig = field(default_factory=MomentsConfig)

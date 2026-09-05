"""Signal aggregator: computes all controller signals for all modules.

Orchestrates per-module computation of surprise, stability, and repetition
signals, producing a ModuleSignals dataclass for each module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..optim.masked_adamw import MaskedAdamW
from .config import ControllerConfig
from .module_index import ModuleId, ModuleInfo
from .moments import SignalMomentTracker
from .repetition import FusedRepetition
from .stability import StabilityTracker
from .surprise import AdamSurprise


@dataclass
class ModuleSignals:
    """All controller signals for a single module at step t."""

    module_id: ModuleId
    grad_norm: float = 0.0  # r_t(j) = ||g_t(j)||₂
    surprise: float = 0.0  # S_t(j)
    stability_C: float = 0.0  # directional cosine (instantaneous)
    stability_C_sustained: float = 0.0  # C̄: EMA of the directional cosine
    stability_adam: float = 0.0  # Adam ratio: mean m²/(v+ε), in [0, 1]
    # MoLF's expected preconditioned descent per parameter (Eq. 4 without
    # the learning rate): mean m²/(√v+ε). Units of loss decrease per
    # parameter; NOT scale-invariant, by design.
    epd_score: float = 0.0
    stability_V: float = 0.0  # windowed variance
    repetition: float = 0.0  # fused R_t(j)
    repetition_components: dict[str, float] = field(default_factory=dict)


class SignalComputer:
    """Computes all controller signals for all modules.

    Honors ``config.ablation``: a disabled signal is not computed (its
    tracker state does not update) and is pinned to its neutral value —
    see :class:`~hivemind.controller.config.AblationConfig` for the
    degraded-policy semantics of each neutral.
    """

    def __init__(self, config: ControllerConfig, modules: list[ModuleInfo]) -> None:
        self.config = config
        self.modules = modules

        ab = config.ablation
        self._surprise_disabled = ab.is_disabled("surprise")
        self._repetition_disabled = ab.is_disabled("repetition")
        self._stability_c_disabled = ab.is_disabled("stability_C")
        self._stability_v_disabled = ab.is_disabled("stability_V")

        pol = config.policy
        self._neutral_surprise = (
            ab.neutral_surprise if ab.neutral_surprise is not None else pol.surprise_high
        )
        self._neutral_repetition = (
            ab.neutral_repetition
            if ab.neutral_repetition is not None
            else pol.repetition_medium
        )
        self._neutral_c = (
            ab.neutral_stability_C
            if ab.neutral_stability_C is not None
            else pol.stability_high_C
        )
        self._neutral_v = (
            ab.neutral_stability_V
            if ab.neutral_stability_V is not None
            else pol.stability_low_V
        )

        # Disabled repetition components: zero their fusion weight; the
        # remaining λs renormalize inside FusedRepetition.
        rep_cfg = config.repetition
        if any(
            ab.is_disabled(f"repetition_{comp}") for comp in ("mom", "hash", "ret")
        ):
            from dataclasses import replace

            rep_cfg = replace(
                rep_cfg,
                lambda_mom=0.0 if ab.is_disabled("repetition_mom") else rep_cfg.lambda_mom,
                lambda_hash=0.0 if ab.is_disabled("repetition_hash") else rep_cfg.lambda_hash,
                lambda_ret=0.0 if ab.is_disabled("repetition_ret") else rep_cfg.lambda_ret,
            )
        self._rep_cfg = rep_cfg

        # Per-module trackers
        self._surprise: dict[ModuleId, AdamSurprise] = {}
        self._stability: dict[ModuleId, StabilityTracker] = {}
        self._repetition: dict[ModuleId, FusedRepetition] = {}

        for m in modules:
            self._surprise[m.id] = AdamSurprise(config.surprise)
            self._stability[m.id] = StabilityTracker(config.stability)
            self._repetition[m.id] = FusedRepetition(rep_cfg)

        # Signal moments (MomentsConfig): the controller's own continuous
        # m̃/ṽ, keyed by parameter name so they checkpoint and resume.
        mc = config.moments
        self._use_tracked = mc.source == "tracked"
        # ``param_names`` are not unique (both LoRA factors of a projection
        # share one name), so suffix repeats: the tracker keys buffers by
        # this string and two params must never share one.
        self._param_names: dict[int, str] = {}
        seen: dict[str, int] = {}
        for m in modules:
            for p, name in zip(m.params, m.param_names):
                k = seen.get(name, 0)
                seen[name] = k + 1
                self._param_names[id(p)] = name if k == 0 else f"{name}#{k}"
        self._moments = SignalMomentTracker(
            sketch_stride=int(mc.sketch_stride), bias_correct=bool(mc.bias_correct)
        )

    @property
    def moments(self) -> SignalMomentTracker:
        return self._moments

    def compute_all(
        self,
        optimizer: MaskedAdamW,
        modules: list[ModuleInfo],
        bucket_id: int = 0,
        embedding: torch.Tensor | None = None,
    ) -> dict[ModuleId, ModuleSignals]:
        """Compute signals for all modules.

        Must be called AFTER backward() but BEFORE optimizer.step().

        Args:
            optimizer: the MaskedAdamW optimizer (to read m/v states).
            modules: list of ModuleInfo with current gradients.
            bucket_id: sample bucket hash (teacher assignment).
            embedding: [d] routing embedding for retrieval repetition.

        Returns:
            {ModuleId: ModuleSignals} for each module.
        """
        results: dict[ModuleId, ModuleSignals] = {}
        tracked = self._use_tracked
        if tracked:
            self._moments.set_betas_from(optimizer)

        for mod in modules:
            mid = mod.id
            signals = ModuleSignals(module_id=mid)

            # Collect gradients and the moment evidence. ``grads`` (full) feeds
            # grad_norm / Top-M; ``sig_grads`` is what the m/v-based signals
            # see — the full gradient with optimizer moments, the strided
            # sketch with tracked moments.
            grads = []
            sig_grads = []
            m_states = []
            v_states = []
            has_states = True
            pending: list[tuple[str, torch.Tensor]] = []

            for p in mod.params:
                if p.grad is None:
                    continue
                g = p.grad.detach().flatten()
                grads.append(g)
                if tracked:
                    name = self._param_names.get(id(p), f"param@{id(p)}")
                    gs = self._moments.sketch(g)
                    m, v = self._moments.read(name, gs)
                    sig_grads.append(gs)
                    m_states.append(m)
                    v_states.append(v)
                    pending.append((name, gs))
                else:
                    sig_grads.append(g)
                    m, v = optimizer.get_state_for_param(p)
                    if m is not None and v is not None:
                        m_states.append(m.detach().flatten())
                        v_states.append(v.detach().flatten())
                    else:
                        has_states = False

            if not grads:
                results[mid] = signals
                continue

            signals.grad_norm = torch.cat(grads).norm().item()
            grad_cat = torch.cat(sig_grads)

            if has_states and m_states and v_states:
                m_cat = torch.cat(m_states)
                v_cat = torch.cat(v_states)

                # Surprise
                if self._surprise_disabled:
                    signals.surprise = self._neutral_surprise
                else:
                    surprise_tracker = self._surprise.get(mid)
                    if surprise_tracker is None:
                        surprise_tracker = AdamSurprise(self.config.surprise)
                        self._surprise[mid] = surprise_tracker
                    signals.surprise = surprise_tracker.compute(grad_cat, v_cat)

                # Stability
                stability_tracker = self._stability.get(mid)
                if stability_tracker is None:
                    stability_tracker = StabilityTracker(self.config.stability)
                    self._stability[mid] = stability_tracker
                if self._stability_c_disabled:
                    signals.stability_C = self._neutral_c
                    signals.stability_C_sustained = self._neutral_c
                else:
                    signals.stability_C = stability_tracker.compute_directional(
                        grad_cat, m_cat
                    )
                    signals.stability_C_sustained = stability_tracker.sustained_C
                signals.stability_adam = stability_tracker.compute_adam_ratio(m_cat, v_cat)
                signals.epd_score = stability_tracker.compute_epd(m_cat, v_cat)
                if self._stability_v_disabled:
                    signals.stability_V = self._neutral_v
                else:
                    signals.stability_V = stability_tracker.update_windowed_variance(
                        signals.grad_norm
                    )

                # Repetition
                if self._repetition_disabled:
                    signals.repetition = self._neutral_repetition
                else:
                    rep_tracker = self._repetition.get(mid)
                    if rep_tracker is None:
                        rep_tracker = FusedRepetition(self._rep_cfg)
                        self._repetition[mid] = rep_tracker
                    signals.repetition, signals.repetition_components = rep_tracker.compute(
                        grad_cat, m_cat, bucket_id, signals.surprise, embedding
                    )
            else:
                # No optimizer states yet (step 0) — use grad norm only
                if self._stability_v_disabled:
                    signals.stability_V = self._neutral_v
                else:
                    stability_tracker = self._stability.get(mid)
                    if stability_tracker:
                        signals.stability_V = stability_tracker.update_windowed_variance(
                            signals.grad_norm
                        )

            for name, gs in pending:
                self._moments.update(name, gs)
            results[mid] = signals

        if tracked:
            self._moments.advance()
        return results

    def state_dict(self) -> dict:
        def _encode(mid: ModuleId) -> str:
            return f"{mid.layer}|{mid.block_type}|{mid.param_type}"

        return {
            "stability": {_encode(mid): tr.state_dict() for mid, tr in self._stability.items()},
            "repetition": {_encode(mid): tr.state_dict() for mid, tr in self._repetition.items()},
            "moments": self._moments.state_dict() if self._use_tracked else None,
        }

    def load_state_dict(self, state: dict) -> None:
        def _decode(key: str) -> ModuleId:
            layer_s, block, ptype = key.split("|")
            return ModuleId(int(layer_s), block, ptype)

        for key, sub in state.get("stability", {}).items():
            mid = _decode(key)
            if mid in self._stability:
                self._stability[mid].load_state_dict(sub)
        for key, sub in state.get("repetition", {}).items():
            mid = _decode(key)
            if mid in self._repetition:
                self._repetition[mid].load_state_dict(sub)
        moments = state.get("moments")
        if self._use_tracked and moments:
            device = next((p.device for m in self.modules for p in m.params), None)
            self._moments.load_state_dict(moments, device=device)

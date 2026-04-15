"""Signal aggregator: computes all controller signals for all modules.

Orchestrates per-module computation of surprise, stability, and repetition
signals, producing a ModuleSignals dataclass for each module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..optim.masked_adamw import MaskedAdamW
from .config import ControllerConfig
from .module_index import ModuleId, ModuleInfo
from .repetition import FusedRepetition
from .stability import StabilityTracker
from .surprise import AdamSurprise


@dataclass
class ModuleSignals:
    """All controller signals for a single module at step t."""

    module_id: ModuleId
    grad_norm: float = 0.0  # r_t(j) = ||g_t(j)||₂
    surprise: float = 0.0  # S_t(j)
    stability_C: float = 0.0  # directional cosine
    stability_adam: float = 0.0  # Adam ratio
    stability_V: float = 0.0  # windowed variance
    repetition: float = 0.0  # fused R_t(j)
    repetition_components: dict[str, float] = field(default_factory=dict)


class SignalComputer:
    """Computes all controller signals for all modules."""

    def __init__(self, config: ControllerConfig, modules: list[ModuleInfo]) -> None:
        self.config = config
        self.modules = modules

        # Per-module trackers
        self._surprise: dict[ModuleId, AdamSurprise] = {}
        self._stability: dict[ModuleId, StabilityTracker] = {}
        self._repetition: dict[ModuleId, FusedRepetition] = {}

        for m in modules:
            self._surprise[m.id] = AdamSurprise(config.surprise)
            self._stability[m.id] = StabilityTracker(config.stability)
            self._repetition[m.id] = FusedRepetition(config.repetition)

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

        for mod in modules:
            mid = mod.id
            signals = ModuleSignals(module_id=mid)

            # Collect gradients and optimizer states
            grads = []
            m_states = []
            v_states = []
            has_states = True

            for p in mod.params:
                if p.grad is None:
                    continue
                grads.append(p.grad.detach().flatten())
                m, v = optimizer.get_state_for_param(p)
                if m is not None and v is not None:
                    m_states.append(m.detach().flatten())
                    v_states.append(v.detach().flatten())
                else:
                    has_states = False

            if not grads:
                results[mid] = signals
                continue

            grad_cat = torch.cat(grads)
            signals.grad_norm = grad_cat.norm().item()

            if has_states and m_states and v_states:
                m_cat = torch.cat(m_states)
                v_cat = torch.cat(v_states)

                # Surprise
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
                signals.stability_C = stability_tracker.compute_directional(grad_cat, m_cat)
                signals.stability_adam = stability_tracker.compute_adam_ratio(m_cat, v_cat)
                signals.stability_V = stability_tracker.update_windowed_variance(
                    signals.grad_norm
                )

                # Repetition
                rep_tracker = self._repetition.get(mid)
                if rep_tracker is None:
                    rep_tracker = FusedRepetition(self.config.repetition)
                    self._repetition[mid] = rep_tracker
                signals.repetition, signals.repetition_components = rep_tracker.compute(
                    grad_cat, m_cat, bucket_id, signals.surprise, embedding
                )
            else:
                # No optimizer states yet (step 0) — use grad norm only
                stability_tracker = self._stability.get(mid)
                if stability_tracker:
                    signals.stability_V = stability_tracker.update_windowed_variance(
                        signals.grad_norm
                    )

            results[mid] = signals

        return results

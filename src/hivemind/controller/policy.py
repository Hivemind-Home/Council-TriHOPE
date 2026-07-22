"""R/F/P routing policy.

Determines where knowledge should be stored for each selected module:
- R (retrieval): surprising but not recurring → store in external memory
- F (fast/LoRA): recurring but not yet stable → learn quickly in adapters
- P (permanent): recurring and stable → consolidate to base weights
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import PolicyConfig
from .module_index import ModuleId
from .signals import ModuleSignals


@dataclass(frozen=True)
class StoreAction:
    """A routing action for a single module."""

    module_id: ModuleId
    store: str  # "R", "F", or "P"
    # Diagnostic fields for the event trace
    surprise: float = 0.0
    repetition: float = 0.0
    stability_C: float = 0.0
    stability_V: float = 0.0
    stability_adam: float = 0.0
    grad_norm: float = 0.0
    repetition_components: dict[str, float] = field(default_factory=dict)


class RFPPolicy:
    """R/F/P routing policy based on surprise, repetition, and stability signals."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def decide(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
    ) -> list[StoreAction]:
        """Determine R/F/P routing for each selected module.

        Steps:
        1. Select Top-M modules by gradient norm.
        2. For each selected module, apply R/F/P policy.

        Args:
            module_signals: {ModuleId: ModuleSignals} for all modules.

        Returns:
            List of StoreAction for selected modules.
        """
        cfg = self.config

        # Step 1: Select Top-M modules by gradient norm
        sorted_modules = sorted(
            module_signals.values(),
            key=lambda s: s.grad_norm,
            reverse=True,
        )
        selected = sorted_modules[: cfg.top_m_modules]

        # Step 2: Apply R/F/P policy to each selected module
        actions: list[StoreAction] = []
        for sig in selected:
            if sig.grad_norm == 0.0:
                continue  # skip modules with no gradient

            store = self._classify(sig)
            actions.append(StoreAction(
                module_id=sig.module_id,
                store=store,
                surprise=sig.surprise,
                repetition=sig.repetition,
                stability_C=sig.stability_C,
                stability_V=sig.stability_V,
                stability_adam=sig.stability_adam,
                grad_norm=sig.grad_norm,
                repetition_components=dict(sig.repetition_components or {}),
            ))

        return actions

    def _classify(self, sig: ModuleSignals) -> str:
        """Classify a module's routing target based on its signals.

        R: high surprise, low repetition → novel one-off
        F: moderate+ surprise, medium+ repetition, stability not proven → LoRA
        P: high repetition, high directional stability, low volatility → base weights
        """
        cfg = self.config
        S = sig.surprise
        R = sig.repetition
        C = sig.stability_C
        V = sig.stability_V

        # P: recurring AND stable
        if R >= cfg.repetition_medium and C >= cfg.stability_high_C and V <= cfg.stability_low_V:
            return "P"

        # R: surprising but not recurring
        if S >= cfg.surprise_high and R < cfg.repetition_low:
            return "R"

        # F: default — recurring or moderate surprise, not stable enough for P
        return "F"

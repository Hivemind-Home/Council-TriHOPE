"""R/F/P routing policy.

Determines where knowledge should be stored for each selected module:
- R (retrieval): surprising but not recurring → store in external memory
- F (fast/LoRA): recurring but not yet stable → learn quickly in adapters
- P (permanent): recurring and stable → consolidate to base weights
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AblationConfig, PolicyConfig
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

    def __init__(
        self,
        config: PolicyConfig | None = None,
        ablation: AblationConfig | None = None,
        override: str | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.ablation = ablation or AblationConfig()
        # Debug short-circuit: "always_p" | "always_f" | "always_r" replaces
        # classification (and skips the confidence gate) — diagnostics fields
        # are still populated so the trace stays informative.
        self.override = override

    def decide(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
        teacher_confidence: float | None = None,
    ) -> list[StoreAction]:
        """Determine R/F/P routing for each selected module.

        Steps:
        1. Select Top-M modules by gradient norm.
        2. For each selected module, apply R/F/P policy.
        3. Optionally gate on teacher confidence: a low-confidence teacher
           should not be allowed to write permanent memory (``no_p``) or
           parameters at all (``force_r``).

        Args:
            module_signals: {ModuleId: ModuleSignals} for all modules.
            teacher_confidence: mean confidence of the supervising teacher
                for this batch (``None`` when unavailable).

        Returns:
            List of StoreAction for selected modules.
        """
        cfg = self.config
        ab = self.ablation
        top_m = ab.top_m_override if ab.top_m_override is not None else cfg.top_m_modules

        # Step 1: Select Top-M modules by gradient norm
        sorted_modules = sorted(
            module_signals.values(),
            key=lambda s: s.grad_norm,
            reverse=True,
        )
        selected = sorted_modules[:top_m]

        low_confidence = (
            ab.use_teacher_confidence
            and teacher_confidence is not None
            and teacher_confidence < ab.confidence_threshold
        )

        # Step 2: Apply R/F/P policy to each selected module
        actions: list[StoreAction] = []
        for sig in selected:
            if sig.grad_norm == 0.0:
                continue  # skip modules with no gradient

            if self.override is not None:
                store = self.override.removeprefix("always_").upper()
            else:
                store = self._classify(sig)

                # Step 3: teacher-confidence gate
                if low_confidence:
                    if ab.confidence_gate == "force_r" and "R" not in ab.disable_stores:
                        store = "R"
                    elif store == "P":
                        store = "F"
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
        disabled_stores = self.ablation.disable_stores
        S = sig.surprise
        R = sig.repetition
        C = sig.stability_C
        V = sig.stability_V

        # P: recurring AND stable
        if (
            "P" not in disabled_stores
            and R >= cfg.repetition_medium
            and C >= cfg.stability_high_C
            and V <= cfg.stability_low_V
        ):
            return "P"

        # R: surprising but not recurring
        if (
            "R" not in disabled_stores
            and S >= cfg.surprise_high
            and R < cfg.repetition_low
        ):
            return "R"

        # F: default — recurring or moderate surprise, not stable enough for P
        return "F"

"""R/F/P routing policy.

Determines where knowledge should be stored for each selected module:
- R (retrieval): surprising but not recurring → store in external memory
- F (fast/LoRA): recurring but not yet stable → learn quickly in adapters
- P (permanent): recurring and stable → consolidate to base weights
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .config import AblationConfig, DebugConfig, PolicyConfig
from .module_index import ModuleId
from .signals import ModuleSignals

_STORES = ("R", "F", "P")


def load_random_shares(
    path: "str | Path | None", spec_id: str
) -> dict[str, tuple[float, float, float]]:
    """Per-phase ``(p_R, p_F, p_P)`` from an ``action_share_by_phase.csv``.

    The CSV is the seed-aggregated table written by ``analysis.run_report``
    (columns ``spec_id, phase, r_share_mean, f_share_mean, p_share_mean``,
    in percent). Rows are filtered to ``spec_id`` and renormalised to sum
    to one. Returns ``{}`` (→ uniform draws) when the file is missing or
    holds no row for the spec, with one warning.
    """
    if not path:
        warnings.warn(
            "random_matched: no random_shares_path given; drawing uniform R/F/P.",
            stacklevel=2,
        )
        return {}
    p = Path(path)
    if not p.exists():
        warnings.warn(
            f"random_matched: shares file {p} not found; drawing uniform R/F/P.",
            stacklevel=2,
        )
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("spec_id", "")) != spec_id:
                continue
            vals = []
            for key in ("r_share", "f_share", "p_share"):
                raw = row.get(f"{key}_mean", row.get(key, ""))
                try:
                    vals.append(max(0.0, float(raw)))
                except (TypeError, ValueError):
                    vals.append(0.0)
            total = sum(vals)
            if total <= 0:
                continue
            out[str(row["phase"])] = (vals[0] / total, vals[1] / total, vals[2] / total)
    if not out:
        warnings.warn(
            f"random_matched: no rows for spec_id={spec_id!r} in {p}; "
            "drawing uniform R/F/P.",
            stacklevel=2,
        )
    return out


@dataclass(frozen=True)
class StoreAction:
    """A routing action for a single module."""

    module_id: ModuleId
    store: str  # "R", "F", or "P"
    # Diagnostic fields for the event trace
    surprise: float = 0.0
    repetition: float = 0.0
    stability_C: float = 0.0
    stability_C_sustained: float = 0.0
    stability_V: float = 0.0
    stability_adam: float = 0.0
    grad_norm: float = 0.0
    repetition_components: dict[str, float] = field(default_factory=dict)
    # Gradient-routing (``mode="teacher_partition"``): ``(teacher_index,
    # num_teachers)`` — the writer opens only that teacher's rank slice.
    teacher_slot: tuple[int, int] | None = None


class RFPPolicy:
    """R/F/P routing policy based on surprise, repetition, and stability signals."""

    def __init__(
        self,
        config: PolicyConfig | None = None,
        ablation: AblationConfig | None = None,
        override: str | None = None,
        *,
        debug: DebugConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or PolicyConfig()
        self.ablation = ablation or AblationConfig()
        # Debug short-circuit: "always_p" | "always_f" | "always_r" replaces
        # classification (and skips the confidence gate) — diagnostics fields
        # are still populated so the trace stays informative.
        # "random_matched" draws the label from per-phase shares instead.
        self.override = override
        self.debug = debug or DebugConfig()
        self.seed = int(seed)
        self._random_shares: dict[str, tuple[float, float, float]] = {}
        if override in ("random_matched", "random_commit"):
            self._random_shares = load_random_shares(
                self.debug.random_shares_path, self.debug.random_shares_spec_id
            )
        if override == "random_commit" and not self._random_shares:
            # random_matched degrades to uniform draws, but random_commit would
            # silently never commit -- a different arm. Fail at startup instead.
            raise ValueError(
                "policy_override=random_commit needs per-phase shares: "
                f"{self.debug.random_shares_path!r} is missing or has no rows for "
                f"spec_id={self.debug.random_shares_spec_id!r}"
            )

    def decide(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
        teacher_confidence: float | None = None,
        *,
        step: int | None = None,
        phase: str | None = None,
        teacher_name: str | None = None,
        teacher_index: int | None = None,
        num_teachers: int | None = None,
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
            step: current training step (only the ``random_matched``
                override reads it — it seeds the draw).
            phase: current stream phase name (selects the share row for
                ``random_matched``; ``None`` → uniform).
            teacher_name: the batch's (rank-synced) teacher; a P action is
                demoted to F while it is in ``debug.block_p_for_teachers``.
            teacher_index / num_teachers: the batch teacher's registry
                index and the registry size — ``teacher_partition`` mode
                routes on the label alone.

        Returns:
            List of StoreAction for selected modules.
        """
        ab = self.ablation

        if self.config.mode == "teacher_partition":
            return self._teacher_partition(module_signals, teacher_index, num_teachers)
        if self.config.mode == "adam_score" and self.config.adam_score_rule == "epd_argmax":
            return self._epd_argmax(module_signals, teacher_name)

        # Step 1: Select Top-M modules by gradient norm
        selected = self.select_top_m(module_signals)

        low_confidence = (
            ab.use_teacher_confidence
            and teacher_confidence is not None
            and teacher_confidence < ab.confidence_threshold
        )

        # Step 2: Apply R/F/P policy to each selected module
        actions: list[StoreAction] = []
        for rank, sig in enumerate(selected):
            if self.override == "random_matched":
                store = self._random_store(step or 0, phase, rank)
            elif self.override == "random_commit":
                # Commit-selection control: classify exactly as the policy
                # does (so defers, and the replay they trigger, are the
                # policy's own), then withhold the policy's commits and commit
                # random non-deferred blocks at the reference's per-phase
                # commit share. Only WHICH blocks are committed changes.
                store = self._classify(sig)
                if store == "P":
                    store = "F"
                if store != "R" and self._random_commit_draw(step or 0, phase, rank):
                    store = "P"
                if low_confidence and store == "P":
                    store = "F"
            elif self.override is not None:
                store = self.override.removeprefix("always_").upper()
            else:
                store = self._classify(sig)

                # Step 3: teacher-confidence gate
                if low_confidence:
                    if ab.confidence_gate == "force_r" and "R" not in ab.disable_stores:
                        store = "R"
                    elif store == "P":
                        store = "F"
            # Selective rollback: a blocked teacher may still write fast
            # weights, never permanent ones (applies to overrides too).
            if (
                store == "P"
                and teacher_name is not None
                and teacher_name in self.debug.block_p_for_teachers
            ):
                store = "F"
            actions.append(StoreAction(
                module_id=sig.module_id,
                store=store,
                surprise=sig.surprise,
                repetition=sig.repetition,
                stability_C=sig.stability_C,
                stability_C_sustained=sig.stability_C_sustained,
                stability_V=sig.stability_V,
                stability_adam=sig.stability_adam,
                grad_norm=sig.grad_norm,
                repetition_components=dict(sig.repetition_components or {}),
            ))

        return actions

    def select_top_m(
        self, module_signals: dict[ModuleId, ModuleSignals]
    ) -> list[ModuleSignals]:
        """Top-M modules by gradient norm (modules with no gradient dropped).

        Pure function of the signals — shared by ``decide`` and by the
        R-store replay micro-step, which must pick modules without touching
        any stateful tracker.
        """
        cfg = self.config
        ab = self.ablation
        top_m = ab.top_m_override if ab.top_m_override is not None else cfg.top_m_modules
        sorted_modules = sorted(
            module_signals.values(),
            key=lambda s: s.grad_norm,
            reverse=True,
        )
        return [s for s in sorted_modules[:top_m] if s.grad_norm != 0.0]

    def _epd_argmax(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
        teacher_name: str | None,
    ) -> list[StoreAction]:
        """MoLF routing: per block, the dense expert (base weights, our P
        module) and the LoRA expert (our F module) compete on
        ``lr_i · epd_score_i``; the winner alone updates. Every block routes
        every step (no Top-M). A block with only one expert (rank 0, or a
        module without gradient) routes to the one it has."""
        cfg = self.config
        disabled = self.ablation.disable_stores
        blocks: dict[tuple[int, str], dict[str, ModuleSignals]] = {}
        for sig in module_signals.values():
            if sig.grad_norm == 0.0:
                continue
            blocks.setdefault((sig.module_id.layer, sig.module_id.block_type), {})[
                sig.module_id.param_type
            ] = sig
        actions: list[StoreAction] = []
        for key in sorted(blocks):
            pair = blocks[key]
            f_sig, p_sig = pair.get("F"), pair.get("P")
            s_f = cfg.epd_lr_lora * f_sig.epd_score if f_sig is not None else None
            s_p = cfg.epd_lr_base * p_sig.epd_score if p_sig is not None else None
            if s_p is not None and "P" not in disabled and (s_f is None or s_p > s_f):
                winner, store = p_sig, "P"
            elif f_sig is not None:
                winner, store = f_sig, "F"
            else:
                continue
            if (
                store == "P"
                and teacher_name is not None
                and teacher_name in self.debug.block_p_for_teachers
            ):
                # blocked teacher: the LoRA expert takes the step instead
                if f_sig is None:
                    continue
                winner, store = f_sig, "F"
            actions.append(StoreAction(
                module_id=winner.module_id,
                store=store,
                surprise=winner.surprise,
                repetition=winner.repetition,
                stability_C=winner.stability_C,
                stability_C_sustained=winner.stability_C_sustained,
                stability_V=winner.stability_V,
                stability_adam=winner.stability_adam,
                grad_norm=winner.grad_norm,
                repetition_components=dict(winner.repetition_components or {}),
            ))
        return actions

    def _teacher_partition(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
        teacher_index: int | None,
        num_teachers: int | None,
    ) -> list[StoreAction]:
        """Gradient Routing: every F-type module opens the batch teacher's
        rank slice. No Top-M, no signals in the decision, no P, no R."""
        if teacher_index is None or not num_teachers:
            raise ValueError(
                "policy.mode=teacher_partition needs teacher_index and num_teachers"
            )
        slot = (int(teacher_index), int(num_teachers))
        actions: list[StoreAction] = []
        for sig in sorted(
            module_signals.values(),
            key=lambda s: (s.module_id.layer, s.module_id.block_type, s.module_id.param_type),
        ):
            if sig.module_id.param_type != "F" or sig.grad_norm == 0.0:
                continue
            actions.append(StoreAction(
                module_id=sig.module_id,
                store="F",
                surprise=sig.surprise,
                repetition=sig.repetition,
                stability_C=sig.stability_C,
                stability_C_sustained=sig.stability_C_sustained,
                stability_V=sig.stability_V,
                stability_adam=sig.stability_adam,
                grad_norm=sig.grad_norm,
                repetition_components=dict(sig.repetition_components or {}),
                teacher_slot=slot,
            ))
        return actions

    def _random_commit_draw(self, step: int, phase: str | None, module_rank: int) -> bool:
        """Bernoulli commit draw for ``policy_override="random_commit"``.

        Probability ``p_P / (p_F + p_P)`` from the reference's per-phase
        shares, so that among non-deferred decisions the commit share matches
        the reference's overall commit share. Seeded like ``_random_store``
        (salted differently), identical on every rank and bit-exact under
        resume; ``disable_stores=[P]`` suppresses it.
        """
        if "P" in self.ablation.disable_stores:
            return False
        shares = self._random_shares.get(phase or "") if self._random_shares else None
        if shares is None:
            return False
        _p_r, p_f, p_p = shares
        if p_f + p_p <= 0:
            return False
        gen = torch.Generator().manual_seed(
            (self.seed * 1_000_033 + int(step) * 104_729 + module_rank * 31 + 17) % (2**63 - 1)
        )
        return bool(torch.rand(1, generator=gen).item() < p_p / (p_f + p_p))

    def _random_store(self, step: int, phase: str | None, module_rank: int) -> str:
        """Budget-matched random label (``policy_override="random_matched"``).

        A fresh ``torch.Generator`` per draw, seeded from ``(seed, step[,
        module_rank])``: identical on every rank, no global-RNG consumption,
        bit-exact under resume. ``disable_stores`` still applies (R → F,
        P → F) so the control never writes where its reference could not.
        """
        shares = self._random_shares.get(phase or "") if self._random_shares else None
        p = shares if shares is not None else (1 / 3, 1 / 3, 1 / 3)
        salt = module_rank if self.debug.random_unit == "module" else 0
        gen = torch.Generator().manual_seed(
            (self.seed * 1_000_003 + int(step) * 7919 + salt) % (2**63 - 1)
        )
        u = torch.rand(1, generator=gen).item()
        acc = 0.0
        store = "F"
        for label, prob in zip(_STORES, p):
            acc += prob
            if u < acc:
                store = label
                break
        if store in self.ablation.disable_stores:
            store = "F"
        return store

    def _classify(self, sig: ModuleSignals) -> str:
        """Classify a module's routing target based on its signals.

        ``mode="rfp"`` (TriHOPE):
        R: high surprise, low repetition → novel one-off
        F: moderate+ surprise, medium+ repetition, stability not proven → LoRA
        P: high repetition, high directional stability, low volatility → base weights

        ``mode="surprise_only"`` (Titans-style): R iff S ≥ surprise_high, else F.
        ``mode="adam_score"`` (MoLF-style): P iff m²/v ≥ adam_score_high, else F.
        """
        cfg = self.config
        disabled_stores = self.ablation.disable_stores
        S = sig.surprise
        R = sig.repetition

        if cfg.mode == "surprise_only":
            if "R" not in disabled_stores and S >= cfg.surprise_high:
                return "R"
            return "F"
        if cfg.mode == "adam_score":
            if "P" not in disabled_stores and sig.stability_adam >= cfg.adam_score_high:
                return "P"
            return "F"
        # P reads either the instantaneous cosine or its sustained EMA C̄
        # (PolicyConfig.stability_source); R and F never look at C.
        C = (
            sig.stability_C_sustained
            if cfg.stability_source == "sustained"
            else sig.stability_C
        )
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

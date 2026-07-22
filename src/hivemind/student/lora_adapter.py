"""LoRA adapter protocol — uniform view over multiple LoRA implementations.

The codebase originally talked directly to ``LoRALinear`` (our hand-rolled
class). When we added an Unsloth/PEFT backbone, the LoRA layout changed:
PEFT's ``lora.Linear`` exposes its base as ``module.base_layer.weight``
and its adapters as ``module.lora_A["default"].weight`` /
``module.lora_B["default"].weight`` (a ``ModuleDict`` keyed by adapter
name). Reaching into both shapes from every consumer would scatter
``isinstance`` checks across the controller, stores, regularizer, and
evaluator.

This module provides a single ``LoRAAdapter`` view per LoRA-shaped
``nn.Module``. The controller / writer / stores work against the
adapter; the adapter knows how to translate to the underlying
implementation's attributes. Two concrete adapters today:

* ``_NativeAdapter`` — wraps ``hivemind.student.lora.LoRALinear``
* ``_PEFTAdapter``    — wraps ``peft.tuners.lora.layer.Linear``
                        (also matches HF's ``peft`` 0.10+ structure)

Adding a third LoRA implementation later is a single file change here.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .lora import LoRALinear

# -- protocol ---------------------------------------------------------------


@runtime_checkable
class LoRAAdapter(Protocol):
    """Stable surface every LoRA implementation must expose."""

    @property
    def module(self) -> nn.Module: ...

    @property
    def base_weight(self) -> nn.Parameter: ...

    @property
    def lora_a(self) -> nn.Parameter: ...

    @property
    def lora_b(self) -> nn.Parameter: ...

    @property
    def scaling(self) -> float: ...

    @property
    def base_params(self) -> list[nn.Parameter]: ...

    @property
    def lora_params(self) -> list[nn.Parameter]: ...

    @property
    def rank(self) -> int: ...

    @property
    def out_features(self) -> int: ...

    def merge_lora_into_base(self) -> None: ...

    def reset_lora(self) -> None: ...

    def get_top_k_mask(
        self,
        k: int,
        granularity: Literal["rank_components", "rows"] = "rank_components",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# -- detection --------------------------------------------------------------


def _is_peft_lora(module: nn.Module) -> bool:
    """Duck-type check for ``peft.tuners.lora.layer.Linear``.

    We deliberately avoid importing ``peft`` here so the file works in
    environments without it. The PEFT layer's distinguishing feature is
    a ``ModuleDict`` named ``lora_A`` plus a ``base_layer`` attribute.
    """
    if not hasattr(module, "base_layer"):
        return False
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)
    if lora_A is None or lora_B is None:
        return False
    return isinstance(lora_A, nn.ModuleDict) and isinstance(lora_B, nn.ModuleDict)


def _peft_default_key(module: nn.Module) -> str:
    """Pick the default adapter name from a PEFT module's ``lora_A`` dict.

    Conventionally this is ``"default"``. We fall back to the first key
    if a non-default adapter name was used.
    """
    keys = list(module.lora_A.keys())  # type: ignore[union-attr]
    if "default" in keys:
        return "default"
    if not keys:
        raise RuntimeError("PEFT LoRA module has no adapters registered.")
    return keys[0]


def get_adapter(module: nn.Module) -> Optional[LoRAAdapter]:
    """Return a ``LoRAAdapter`` view of ``module``, or ``None`` if not LoRA."""
    if isinstance(module, LoRALinear):
        return _NativeAdapter(module)
    if _is_peft_lora(module):
        return _PEFTAdapter(module)
    return None


def iter_lora_adapters(model: nn.Module) -> list[tuple[str, LoRAAdapter]]:
    """Yield ``(qualified_name, adapter)`` for every LoRA-capable submodule."""
    out: list[tuple[str, LoRAAdapter]] = []
    for name, submod in model.named_modules():
        adapter = get_adapter(submod)
        if adapter is not None:
            out.append((name, adapter))
    return out


# -- native ----------------------------------------------------------------


class _NativeAdapter:
    """Trivial pass-through to our in-repo ``LoRALinear``."""

    def __init__(self, m: LoRALinear) -> None:
        self._m = m

    @property
    def module(self) -> nn.Module:
        return self._m

    @property
    def base_weight(self) -> nn.Parameter:
        return self._m.base.weight

    @property
    def lora_a(self) -> nn.Parameter:
        return self._m.lora_A

    @property
    def lora_b(self) -> nn.Parameter:
        return self._m.lora_B

    @property
    def scaling(self) -> float:
        return float(self._m.scaling)

    @property
    def base_params(self) -> list[nn.Parameter]:
        return self._m.base_params

    @property
    def lora_params(self) -> list[nn.Parameter]:
        return self._m.lora_params

    @property
    def rank(self) -> int:
        return int(self._m.rank)

    @property
    def out_features(self) -> int:
        return int(self._m.out_features)

    def merge_lora_into_base(self) -> None:
        self._m.merge_lora_into_base()

    def reset_lora(self) -> None:
        self._m.reset_lora()

    def get_top_k_mask(self, k, granularity="rank_components"):
        return self._m.get_top_k_mask(k, granularity=granularity)


# -- PEFT -------------------------------------------------------------------


class _PEFTAdapter:
    """Adapter for ``peft.tuners.lora.layer.Linear`` (PEFT >= 0.10)."""

    def __init__(self, m: nn.Module) -> None:
        self._m = m
        self._key = _peft_default_key(m)

    @property
    def module(self) -> nn.Module:
        return self._m

    @property
    def _lora_A_layer(self) -> nn.Linear:
        return self._m.lora_A[self._key]  # type: ignore[index]

    @property
    def _lora_B_layer(self) -> nn.Linear:
        return self._m.lora_B[self._key]  # type: ignore[index]

    @property
    def base_weight(self) -> nn.Parameter:
        return self._m.base_layer.weight  # type: ignore[union-attr]

    @property
    def lora_a(self) -> nn.Parameter:
        return self._lora_A_layer.weight

    @property
    def lora_b(self) -> nn.Parameter:
        return self._lora_B_layer.weight

    @property
    def scaling(self) -> float:
        s = getattr(self._m, "scaling", None)
        if isinstance(s, dict):
            return float(s.get(self._key, 1.0))
        if s is None:
            return 1.0
        return float(s)

    @property
    def base_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self._m.base_layer.weight]  # type: ignore[union-attr]
        bias = getattr(self._m.base_layer, "bias", None)  # type: ignore[union-attr]
        if isinstance(bias, nn.Parameter):
            params.append(bias)
        return params

    @property
    def lora_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.lora_a, self.lora_b]
        # Some PEFT versions add an embedding-style alpha; ignore unless
        # it's a true Parameter we should include in the F-store group.
        return params

    @property
    def rank(self) -> int:
        # PEFT stores rank as the in_features of lora_B (== out_features of lora_A).
        return int(self.lora_a.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.lora_b.shape[0])

    def merge_lora_into_base(self) -> None:
        # Prefer PEFT's own merge if available — it handles dropout / dtype
        # nuances that our fallback doesn't.
        merge_fn = getattr(self._m, "merge", None)
        if callable(merge_fn):
            try:
                merge_fn(safe_merge=False, adapter_names=[self._key])
                # PEFT merges in-place; reset our LoRA view to zero so
                # downstream "merge then reset" semantics still hold.
                self.reset_lora()
                return
            except TypeError:
                # older PEFT signatures (no kwargs)
                try:
                    merge_fn()
                    self.reset_lora()
                    return
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: hand-compute Δ = B @ A · scaling and add to base.
        with torch.no_grad():
            delta = (self.lora_b @ self.lora_a) * self.scaling
            self.base_weight.add_(delta.to(self.base_weight.dtype))
        self.reset_lora()

    def reset_lora(self) -> None:
        # Standard LoRA init: kaiming on A, zeros on B → ΔW = 0 at start.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def get_top_k_mask(
        self,
        k: int,
        granularity: Literal["rank_components", "rows"] = "rank_components",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.lora_a
        b = self.lora_b

        if a.grad is None or b.grad is None:
            return (
                torch.ones_like(a, dtype=torch.bool),
                torch.ones_like(b, dtype=torch.bool),
            )

        if granularity == "rank_components":
            scores = b.grad.norm(dim=0) + a.grad.norm(dim=1)  # [rank]
            k_clamped = min(k, self.rank)
            _, top_indices = scores.topk(k_clamped)
            mask_A = torch.zeros(self.rank, dtype=torch.bool, device=a.device)
            mask_B = torch.zeros(self.rank, dtype=torch.bool, device=b.device)
            mask_A[top_indices] = True
            mask_B[top_indices] = True
            mask_A = mask_A.unsqueeze(1).expand_as(a)
            mask_B = mask_B.unsqueeze(0).expand_as(b)
            return mask_A, mask_B

        if granularity == "rows":
            row_scores = b.grad.norm(dim=1)  # [out_features]
            k_clamped = min(k, self.out_features)
            _, top_indices = row_scores.topk(k_clamped)
            mask_B = torch.zeros(self.out_features, dtype=torch.bool, device=b.device)
            mask_B[top_indices] = True
            mask_B = mask_B.unsqueeze(1).expand_as(b)
            mask_A = torch.ones_like(a, dtype=torch.bool)
            return mask_A, mask_B

        raise ValueError(f"Unknown granularity: {granularity}")

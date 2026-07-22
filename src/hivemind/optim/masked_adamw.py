"""AdamW with exact per-coordinate write masking (paper Theorem 1).

Extends PyTorch's AdamW so that the R/F/P controller's write decisions become
real optimizer boundaries:

1. Exposes ``m_{t-1}`` / ``v_{t-1}`` for controller signal computation.
2. Accepts a per-step authorization mask per parameter. A *closed* coordinate
   receives no parameter update, no moment update, no AMSGrad-max update, no
   decoupled weight decay, and its per-coordinate update counter does not age
   (Theorem 1 / exact state isolation).
3. Bias correction is coordinate-local: an open coordinate uses
   ``1 - beta ** n_i`` where ``n_i`` is that coordinate's own accepted-update
   count (Corollary 1), tracked in ``state["coord_step"]``.
4. ``reset_state_for_params`` zeroes all optimizer state for a parameter set,
   used after LoRA-to-base consolidation (Corollary 2).

With every coordinate open at every call this reduces exactly to ordinary
single-tensor AdamW (Lemma 1) — verified against ``torch.optim.AdamW`` in
``tests/test_masked_adamw.py``.

Memory note: ``coord_step`` is kept as a 0-dim scalar tensor while every mask
the parameter has seen was uniform (fully open / fully closed) and is lazily
materialized to a full tensor on the first partial mask. Base weight matrices
are only ever opened or closed whole, so full counters exist only for the
small LoRA factors.
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW


class MaskedAdamW(AdamW):
    """AdamW with exact per-coordinate state isolation under write masks."""

    #: Sentinel mask values (no tensor materialized for uniform masks).
    FULLY_OPEN = True
    FULLY_CLOSED = False

    _REJECTED_MODES = ("fused", "foreach", "capturable", "differentiable", "maximize")

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        **kwargs: Any,
    ) -> None:
        for mode in self._REJECTED_MODES:
            if kwargs.pop(mode, None):
                raise ValueError(
                    f"MaskedAdamW does not support {mode}=True; only the plain "
                    "single-tensor minimize path has exact-isolation semantics."
                )
        if kwargs:
            raise TypeError(f"Unexpected arguments: {sorted(kwargs)}")
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        self._controller_params: set[nn.Parameter] = set()
        self._staged_masks: dict[nn.Parameter, torch.Tensor | bool] = {}

    # ------------------------------------------------------------------
    # State exposure for the controller (read AFTER backward, BEFORE step)
    # ------------------------------------------------------------------

    def get_state_for_param(
        self, param: nn.Parameter
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(m_{t-1}, v_{t-1})``, or ``(None, None)`` if no state yet."""
        state = self.state.get(param)
        if state is None or len(state) == 0:
            return None, None
        return state.get("exp_avg"), state.get("exp_avg_sq")

    def get_states_for_params(
        self, params: list[nn.Parameter]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get m and v states for a list of parameters, flattened per param."""
        ms, vs = [], []
        for p in params:
            m, v = self.get_state_for_param(p)
            if m is not None and v is not None:
                ms.append(m.detach().flatten())
                vs.append(v.detach().flatten())
        if not ms:
            return [], []
        return ms, vs

    # ------------------------------------------------------------------
    # Mask staging
    # ------------------------------------------------------------------

    def register_controller_params(self, params: Iterable[nn.Parameter]) -> None:
        """Mark parameters whose per-step default is FULLY_CLOSED.

        Controller-indexed parameters update only when an R/F/P action opens
        them via :meth:`set_masks`. Unregistered parameters (shared embeddings,
        norms, lm_head, ...) stay FULLY_OPEN by default and follow the ordinary
        AdamW rule.
        """
        self._controller_params.update(params)

    def set_masks(self, masks: dict[nn.Parameter, torch.Tensor | bool]) -> None:
        """Stage authorization masks for the next ``step()`` call.

        Values may be ``True`` (fully open), ``False`` (fully closed), or a
        0/1 tensor broadcastable to the parameter shape (e.g. ``[r, 1]`` for a
        LoRA A factor of shape ``[r, d_in]``). Masks are consumed and cleared
        by the next ``step()`` — staging works through ``scaler.step(opt)``,
        which calls ``opt.step()`` with no arguments.
        """
        self._staged_masks.update(masks)

    def clear_masks(self) -> None:
        """Drop staged masks (e.g. after a GradScaler-skipped step)."""
        self._staged_masks = {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        try:
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                amsgrad = group["amsgrad"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("MaskedAdamW does not support sparse gradients")

                    if p in self._staged_masks:
                        mask = self._staged_masks[p]
                    elif p in self._controller_params:
                        mask = self.FULLY_CLOSED
                    else:
                        mask = self.FULLY_OPEN

                    if mask is False:
                        # Fully closed: nothing changes, state is not even
                        # lazily initialized (Theorem 1).
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        self._init_state(state, p, amsgrad)

                    if mask is True and state["coord_step"].dim() == 0:
                        self._step_uniform(
                            p, state, lr, beta1, beta2, eps, weight_decay, amsgrad
                        )
                    else:
                        mask_b = (
                            torch.ones_like(p, dtype=torch.bool)
                            if mask is True
                            else mask.to(device=p.device, dtype=torch.bool)
                        )
                        self._step_masked(
                            p, state, mask_b, lr, beta1, beta2, eps, weight_decay, amsgrad
                        )
        finally:
            self._staged_masks = {}

        return loss

    @staticmethod
    def _init_state(state: dict[str, Any], p: nn.Parameter, amsgrad: bool) -> None:
        state["step"] = torch.zeros((), dtype=torch.float32)
        state["coord_step"] = torch.zeros((), dtype=torch.float32, device=p.device)
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @staticmethod
    def _step_uniform(
        p: nn.Parameter,
        state: dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
    ) -> None:
        """All coordinates open, uniform history: exact stock AdamW op order."""
        grad = p.grad
        state["step"] += 1
        state["coord_step"] += 1
        n = float(state["coord_step"].item())

        p.mul_(1 - lr * weight_decay)
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**n
        bias_correction2 = 1 - beta2**n
        if amsgrad:
            max_sq = state["max_exp_avg_sq"]
            torch.maximum(max_sq, exp_avg_sq, out=max_sq)
            denom_src = max_sq
        else:
            denom_src = exp_avg_sq
        denom = (denom_src.sqrt() / (bias_correction2**0.5)).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))

    @staticmethod
    def _step_masked(
        p: nn.Parameter,
        state: dict[str, Any],
        mask_b: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
    ) -> None:
        """Elementwise path: open coords follow AdamW with coordinate-local
        bias correction; closed coords are left bit-identical."""
        grad = p.grad
        state["step"] += 1

        cs = state["coord_step"]
        if cs.dim() == 0:
            cs = torch.full_like(p, float(cs.item()), dtype=torch.float32)
            state["coord_step"] = cs
        cs.add_(mask_b.to(cs.dtype))

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.copy_(torch.where(mask_b, exp_avg * beta1 + grad * (1 - beta1), exp_avg))
        exp_avg_sq.copy_(
            torch.where(mask_b, exp_avg_sq * beta2 + grad * grad * (1 - beta2), exp_avg_sq)
        )
        if amsgrad:
            max_sq = state["max_exp_avg_sq"]
            max_sq.copy_(torch.where(mask_b, torch.maximum(max_sq, exp_avg_sq), max_sq))
            denom_src = max_sq
        else:
            denom_src = exp_avg_sq

        # Never-opened coords have cs == 0 => bias correction would be 0; they
        # are necessarily masked out by `where`, the clamp only avoids inf/nan
        # in the discarded branch.
        tiny = torch.finfo(torch.float32).tiny
        bc1 = (1 - beta1**cs).clamp_min(tiny)
        bc2 = (1 - beta2**cs).clamp_min(tiny)
        denom = denom_src.float().sqrt() / bc2.sqrt() + eps
        update = (lr / bc1) * exp_avg.float() / denom
        new_p = p.float() * (1 - lr * weight_decay) - update
        p.copy_(torch.where(mask_b, new_p.to(p.dtype), p))

    # ------------------------------------------------------------------
    # Consolidation support (Corollary 2)
    # ------------------------------------------------------------------

    def reset_state_for_params(self, params: Iterable[nn.Parameter]) -> None:
        """Zero all optimizer state for the given parameters.

        Called after a LoRA-to-base merge reinitializes the adapter: the next
        accepted update starts from fresh moments with first-update bias
        correction. ``coord_step`` collapses back to a scalar zero.
        """
        for p in params:
            state = self.state.get(p)
            if not state:
                continue
            state["step"].zero_()
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].zero_()
            state["coord_step"] = torch.zeros((), dtype=torch.float32, device=p.device)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state, keeping ``coord_step`` in exact float32.

        Stock ``Optimizer.load_state_dict`` casts every non-``step`` float
        state tensor to the parameter dtype; for bf16 parameters that would
        corrupt integer counts above 256. Counts are stashed and re-attached
        untouched after the parent load.
        """
        saved_coord = {
            pid: pstate["coord_step"]
            for pid, pstate in state_dict.get("state", {}).items()
            if isinstance(pstate, dict) and "coord_step" in pstate
        }
        super().load_state_dict(state_dict)
        if saved_coord:
            id_map = dict(
                zip(
                    chain.from_iterable(g["params"] for g in state_dict["param_groups"]),
                    chain.from_iterable(g["params"] for g in self.param_groups),
                )
            )
            for old_id, coord in saved_coord.items():
                param = id_map[old_id]
                self.state[param]["coord_step"] = (
                    coord.detach().clone().to(device=param.device, dtype=torch.float32)
                )

"""AdamW with exposed optimizer states and gradient masking.

Extends PyTorch's AdamW to:
1. Expose m_{t-1} and v_{t-1} for controller signal computation.
2. Support gradient masking for selective (Top-K) parameter updates.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.optim import AdamW


class MaskedAdamW(AdamW):
    """AdamW optimizer with exposed Adam states and gradient masking support."""

    def get_state_for_param(
        self, param: nn.Parameter
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get the Adam first and second moment estimates for a parameter.

        Returns (m_{t-1}, v_{t-1}) — the states BEFORE the current step.
        Returns (None, None) if the parameter has no optimizer state yet.
        """
        state = self.state.get(param)
        if state is None or len(state) == 0:
            return None, None
        m = state.get("exp_avg")
        v = state.get("exp_avg_sq")
        return m, v

    def get_states_for_params(
        self, params: list[nn.Parameter]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get m and v states for a list of parameters, concatenated.

        Returns (m_cat, v_cat) as flat tensors. Params without state are skipped.
        """
        ms, vs = [], []
        for p in params:
            m, v = self.get_state_for_param(p)
            if m is not None and v is not None:
                ms.append(m.detach().flatten())
                vs.append(v.detach().flatten())
        if not ms:
            return [], []
        return ms, vs

    def step_with_mask(
        self,
        gradient_masks: dict[nn.Parameter, torch.Tensor] | None = None,
        closure=None,
    ):
        """Perform a single optimization step, optionally masking gradients.

        Args:
            gradient_masks: {param: mask_tensor} where mask is 0/1.
                           Only parameters in the dict are masked; others update normally.
            closure: optional closure for re-evaluating loss.
        """
        if gradient_masks is not None:
            # Temporarily mask gradients
            saved_grads = {}
            for param, mask in gradient_masks.items():
                if param.grad is not None:
                    saved_grads[param] = param.grad.clone()
                    param.grad.mul_(mask.float())

        loss = super().step(closure)

        if gradient_masks is not None:
            # Restore original gradients (for logging/signal computation)
            for param, orig_grad in saved_grads.items():
                param.grad = orig_grad

        return loss

    def zero_grad_for_params(self, params: list[nn.Parameter]) -> None:
        """Zero gradients for a specific subset of parameters."""
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

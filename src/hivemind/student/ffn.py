"""SwiGLU feed-forward network with LoRA adapters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoRAConfig
from .lora import LoRALinear


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: down(silu(gate(x)) * up(x)).

    LoRA adapters can be attached to gate, up, and down projections.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiplier: int = 4,
        dropout: float = 0.0,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * multiplier
        # SwiGLU uses 2/3 of the hidden dim for gate and up to match param count
        # but for simplicity and following common practice, we use full hidden_dim
        self.hidden_dim = hidden_dim

        targets = set(lora_config.target_modules) if lora_config else set()

        def _make_proj(name: str, in_f: int, out_f: int) -> nn.Module:
            if name in targets and lora_config is not None:
                return LoRALinear(
                    in_f, out_f, rank=lora_config.rank,
                    alpha=lora_config.alpha, dropout=lora_config.dropout, bias=False,
                )
            return nn.Linear(in_f, out_f, bias=False)

        self.w_gate = _make_proj("gate", dim, hidden_dim)
        self.w_up = _make_proj("up", dim, hidden_dim)
        self.w_down = _make_proj("down", hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: down(silu(gate(x)) * up(x))."""
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))

    def get_lora_modules(self) -> dict[str, LoRALinear]:
        """Return LoRA-enabled projections."""
        result = {}
        for name, proj in [("gate", self.w_gate), ("up", self.w_up), ("down", self.w_down)]:
            if isinstance(proj, LoRALinear):
                result[name] = proj
        return result

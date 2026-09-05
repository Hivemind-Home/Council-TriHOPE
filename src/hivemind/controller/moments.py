"""Signal moments: continuous Adam-style m/v kept for the controller.

Why this exists. MaskedAdamW gives Theorem 1 its teeth by never touching
the optimizer state of a closed coordinate: no parameter, moment,
weight-decay or bias-correction change. The controller, however, reads
``m_{t-1}`` / ``v_{t-1}`` as *evidence* — surprise is ``g²/v``, stability
is ``cos(g, m)``. Under exact masking those two requirements collide: a
base-weight module whose coordinates have never been opened has ``m = v =
0`` forever, so its surprise saturates at ``s_max`` (``g²/ε``) and its
cosine is a warmup zero — and since P is the only action that opens base
coordinates, and P requires those very signals, P can never fire. The
first real-stream run showed exactly this: C̄ = 0.000 and S = 20.0 at all
9 384 decisions in 2 000 steps (see ``scripts/diagnose_p.py``).

The fix is MoLF's "universal momentum tracking" (Tang et al., 2026,
Sec. 4.3) confined to the signal path: the controller maintains its own
exponential moments for every indexed coordinate, updated from the full
gradient every step with the optimizer's own β₁/β₂, and bias-corrected.
On a coordinate that is open at every step these are *identical* to
Adam's ``exp_avg`` / ``exp_avg_sq`` (same recursion, same inputs — tested);
on a closed coordinate they keep advancing while the optimizer's copy
stays frozen. The update path is untouched, so Theorem 1 still holds for
what actually moves.

Cost: two fp32 buffers per tracked coordinate. ``sketch_stride = k`` keeps
them for every k-th coordinate only; every signal is a mean or a cosine
over coordinates, so a fixed strided subsample is an unbiased, deterministic
(no RNG, DDP-identical) estimate. For the 0.6B student, stride 8 is
≈ 0.6 GB.
"""

from __future__ import annotations

from typing import Iterable

import torch


class SignalMomentTracker:
    """Per-parameter EMA pair ``(m̃, ṽ)`` for the controller's signals."""

    def __init__(
        self,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        sketch_stride: int = 1,
        bias_correct: bool = True,
    ) -> None:
        if sketch_stride < 1:
            raise ValueError(f"sketch_stride must be >= 1, got {sketch_stride}")
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.stride = int(sketch_stride)
        self.bias_correct = bool(bias_correct)
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._step: int = 0  # completed update rounds
        self._betas_locked = False

    # -- configuration ---------------------------------------------------

    def set_betas_from(self, optimizer: torch.optim.Optimizer) -> None:
        """Adopt the optimizer's β₁/β₂ so open coordinates match Adam exactly."""
        if self._betas_locked:
            return
        groups = [tuple(float(b) for b in g["betas"]) for g in optimizer.param_groups]
        if len(set(groups)) != 1:
            raise ValueError(
                f"SignalMomentTracker needs one (beta1, beta2) across param groups, got {groups}"
            )
        self.beta1, self.beta2 = groups[0]
        self._betas_locked = True

    @property
    def step(self) -> int:
        return self._step

    # -- per-step protocol ---------------------------------------------------

    def sketch(self, flat: torch.Tensor) -> torch.Tensor:
        """Strided view of a flattened tensor (the coordinates we track)."""
        return flat if self.stride == 1 else flat[:: self.stride]

    def read(self, name: str, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-update, bias-corrected ``(m̂_{t-1}, v̂_{t-1})`` for ``name``.

        Zeros (no history) before the first update.
        """
        m = self._m.get(name)
        v = self._v.get(name)
        if m is None or v is None or self._step == 0:
            z = torch.zeros_like(like, dtype=torch.float32)
            return z, z
        if not self.bias_correct:
            return m, v
        t = self._step
        return m / (1.0 - self.beta1**t), v / (1.0 - self.beta2**t)

    def update(self, name: str, grad_sketch: torch.Tensor) -> None:
        """Advance ``name``'s moments with this step's (sketched) gradient."""
        g = grad_sketch.detach().float()
        m = self._m.get(name)
        if m is None or m.shape != g.shape or m.device != g.device:
            m = torch.zeros_like(g)
            self._m[name] = m
            self._v[name] = torch.zeros_like(g)
        v = self._v[name]
        m.lerp_(g, 1.0 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

    def advance(self) -> None:
        """Mark one training step's updates complete (bias-correction clock)."""
        self._step += 1

    # -- persistence -----------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "stride": self.stride,
            "m": {k: t.detach().clone() for k, t in self._m.items()},
            "v": {k: t.detach().clone() for k, t in self._v.items()},
        }

    def load_state_dict(self, state: dict, device: torch.device | None = None) -> None:
        self._step = int(state.get("step", 0))
        if int(state.get("stride", self.stride)) != self.stride:
            raise ValueError(
                f"checkpoint sketch_stride={state.get('stride')} != config {self.stride}; "
                "the tracked coordinates would not line up"
            )
        def _place(t: torch.Tensor) -> torch.Tensor:
            return t.to(device) if device is not None else t

        self._m = {k: _place(t) for k, t in state.get("m", {}).items()}
        self._v = {k: _place(t) for k, t in state.get("v", {}).items()}

    def names(self) -> Iterable[str]:
        return self._m.keys()

"""Interference oracle for the ``schedule_oracle_bad`` / ``schedule_oracle_good``
rebuttal controls (ICLR 2027, reviewer BDmh Q2: "commit ... those with the
largest interference with earlier domains").

These arms commit exactly the reference controller's per-step count ``k_t``
(like ``schedule_random`` / ``schedule_anti``), but rank the non-deferred
candidate blocks by an *oracle* interference score that the controller cannot
see:

    score_b = < g_old_b , Δθ_b >

* ``Δθ_b`` is the update MaskedAdamW would apply to block ``b`` this step if
  the block were opened (current gradient, pre-step m/v, the block's own
  coordinate-local bias correction, decoupled weight decay) — reproduced in
  closed form here, without touching optimizer state.
* ``g_old_b`` is the gradient, w.r.t. block ``b``'s parameters, of the
  retention loss on a probe set of *training-split* rows from every domain
  other than the current phase's domain(s): per-domain per-token CE (the same
  loss the retention eval computes), averaged over domains (as M1 averages).

So ``score_b`` is the first-order predicted change in earlier-domain loss if
``b`` is committed now. ``oracle_bad`` commits the highest scores (most
predicted forgetting), ``oracle_good`` the lowest. Together they bracket how
much *which-block* selection can move forgetting at the controller's volume.

Probe rows come from the training stream only (the first ``rows_per_domain``
rows of each domain as they are first seen). Validation rows are never used.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_index import ModuleId, ModuleInfo

ORACLE_OVERRIDES = ("schedule_oracle_bad", "schedule_oracle_good")
IGNORE_INDEX = -100


class ProbeBuffer:
    """First ``rows_per_domain`` training rows of each domain (CPU, unpadded).

    Rows are stored trimmed to their attended positions, so right- or
    left-padding in the collated batch does not matter. Filled in stream
    order and never replaced, so the content is a deterministic function of
    the data stream; ``state_dict`` lets a resumed run continue with it.
    """

    def __init__(self, rows_per_domain: int) -> None:
        if rows_per_domain <= 0:
            raise ValueError("rows_per_domain must be positive")
        self.rows_per_domain = int(rows_per_domain)
        self.rows: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        # Step each row was captured at (parallel to ``rows``), so a resumed
        # run can drop rows captured after its checkpoint and re-capture them
        # identically instead of duplicating them.
        self.steps: dict[str, list[int]] = {}

    def add_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        domains: Iterable[str] | None,
        step: int = -1,
    ) -> bool:
        """Store rows of not-yet-full domains. Returns True if anything was added."""
        if domains is None:
            return False
        domains = list(domains)
        if labels is None:
            labels = input_ids
        changed = False
        for b, dom in enumerate(domains):
            if not dom or b >= input_ids.shape[0]:
                continue
            if len(self.rows.get(str(dom), [])) >= self.rows_per_domain:
                continue
            keep = (
                attention_mask[b].bool()
                if attention_mask is not None
                else torch.ones_like(input_ids[b], dtype=torch.bool)
            )
            ids = input_ids[b][keep].detach().to("cpu").clone()
            lab = labels[b][keep].detach().to("cpu").clone()
            if ids.numel() < 2 or not bool((lab[1:] != IGNORE_INDEX).any()):
                continue  # nothing to score on
            self.rows.setdefault(str(dom), []).append((ids, lab))
            self.steps.setdefault(str(dom), []).append(int(step))
            changed = True
        return changed

    def probe(self, exclude: Iterable[str]) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Rows of every buffered domain not in ``exclude`` (sorted by domain)."""
        ex = {str(d) for d in exclude}
        return {d: list(r) for d, r in sorted(self.rows.items()) if d not in ex and r}

    def state_dict(self) -> dict[str, Any]:
        return {"rows_per_domain": self.rows_per_domain, "rows": self.rows, "steps": self.steps}

    def load_state_dict(self, state: dict[str, Any], before_step: int | None = None) -> None:
        """Restore; with ``before_step``, keep only rows captured before it."""
        self.rows_per_domain = int(state["rows_per_domain"])
        self.rows, self.steps = {}, {}
        for d, r in state["rows"].items():
            steps = state.get("steps", {}).get(d, [-1] * len(r))
            for (a, b), s in zip(r, steps):
                if before_step is not None and s >= before_step:
                    continue
                self.rows.setdefault(str(d), []).append((a.clone(), b.clone()))
                self.steps.setdefault(str(d), []).append(int(s))


def _group_of(optimizer: torch.optim.Optimizer) -> dict[int, dict[str, Any]]:
    return {id(p): g for g in optimizer.param_groups for p in g["params"]}


@torch.no_grad()
def adamw_open_delta(
    p: nn.Parameter, optimizer: torch.optim.Optimizer, group: dict[str, Any]
) -> torch.Tensor | None:
    """The update MaskedAdamW would apply to ``p`` this step if fully opened.

    Mirrors ``MaskedAdamW._step_masked`` with an all-True mask (identical
    math to ``_step_uniform``): m/v updated with the current gradient, the
    coordinate-local step count incremented, bias-corrected, decoupled
    weight decay. Returns ``Δθ`` in fp32, or ``None`` if ``p`` has no grad.
    Optimizer state is only read.
    """
    if p.grad is None:
        return None
    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    wd = float(group.get("weight_decay", 0.0))
    g = p.grad.detach().float()
    st = optimizer.state.get(p, {})
    m = st.get("exp_avg")
    v = st.get("exp_avg_sq")
    m = torch.zeros_like(g) if m is None else m.detach().float()
    v = torch.zeros_like(g) if v is None else v.detach().float()
    cs = st.get("coord_step")
    n = (torch.zeros((), dtype=torch.float32, device=g.device) if cs is None
         else cs.detach().float().to(g.device)) + 1.0
    m1 = m * beta1 + g * (1 - beta1)
    v1 = v * beta2 + g * g * (1 - beta2)
    if group.get("amsgrad", False) and st.get("max_exp_avg_sq") is not None:
        v1 = torch.maximum(st["max_exp_avg_sq"].detach().float(), v1)
    # Bias corrections in float64: 1 - beta2**n cancels badly in fp32 for
    # small n (relative error ~1e-5 at n=5).
    n64 = n.double()
    bc1 = (1 - beta1 ** n64).float()
    bc2_sqrt = (1 - beta2 ** n64).sqrt().float()
    denom = v1.sqrt() / bc2_sqrt + eps
    return -lr * wd * p.detach().float() - (lr / bc1) * m1 / denom


def interference_scores(
    student: nn.Module,
    modules: list[ModuleInfo],
    optimizer: torch.optim.Optimizer,
    probe_rows: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
    autocast_ctx: Any = None,
) -> dict[ModuleId, float]:
    """``{module_id: <g_old, Δθ>}`` for ``modules`` (see module docstring).

    Side-effect free with respect to training: ``.grad`` of every parameter
    is untouched (``torch.autograd.grad``, not ``backward``), optimizer state
    is only read, the CPU/CUDA RNG state is restored afterwards, and the
    module's train/eval mode is restored. Must run after the main backward
    (``.grad`` holds this step's gradient) and before ``optimizer.step()``.
    """
    if not probe_rows or not modules:
        return {}
    groups = _group_of(optimizer)
    params: list[nn.Parameter] = []
    owner: list[ModuleId] = []
    deltas: list[torch.Tensor] = []
    for mod in modules:
        for p in mod.params:
            if not p.requires_grad or id(p) not in groups:
                continue
            d = adamw_open_delta(p, optimizer, groups[id(p)])
            if d is None:
                continue
            params.append(p)
            owner.append(mod.id)
            deltas.append(d)
    scores: dict[ModuleId, float] = {m.id: 0.0 for m in modules}
    if not params:
        return scores

    acc = [torch.zeros_like(d) for d in deltas]
    n_dom = len(probe_rows)
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    was_training = student.training
    rng_devices = [device.index or 0] if device.type == "cuda" else []
    student.eval()
    try:
        with torch.random.fork_rng(devices=rng_devices), torch.enable_grad():
            for _dom, rows in probe_rows.items():
                ntok = sum(int((lab[1:] != IGNORE_INDEX).sum()) for _, lab in rows)
                if ntok == 0:
                    continue
                for ids, lab in rows:
                    ids_d = ids.to(device).unsqueeze(0)
                    lab_d = lab.to(device).unsqueeze(0)
                    with ctx:
                        logits = student(ids_d)
                    shift_logits = logits[:, :-1, :].float()
                    shift_labels = lab_d[:, 1:]
                    tok_sum = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.shape[-1]),
                        shift_labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    )
                    loss = tok_sum / (ntok * n_dom)
                    grads = torch.autograd.grad(loss, params, allow_unused=True)
                    for a, g in zip(acc, grads):
                        if g is not None:
                            a.add_(g.detach().float())
    finally:
        student.train(was_training)

    for mid, a, d in zip(owner, acc, deltas):
        scores[mid] += float((a * d).sum().item())
    return scores

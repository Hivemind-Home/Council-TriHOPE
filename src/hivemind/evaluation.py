"""Evaluation harness for validation splits.

Implements the Theory 101 §9 metrics:

* Distillation quality — per-domain CE loss, perplexity.
* Forgetting — ``loss_current - loss_baseline`` per domain, where the
  baseline is the first evaluation's loss (captured on first call).
  Positive = the model got worse on an older domain; this is the signal
  catastrophic forgetting is happening.
* Memory efficiency — per-LoRA-module sparsity (fraction of
  near-zero entries in the active LoRA components).
* Retrieval reliance — R-store size and hit rate on a fixed query set.

Per-domain metrics are produced by running a separate val dataloader per
domain. The trainer owns dataloader construction (``build_val_loaders``)
so all splits share the same tokenizer + collator as train.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distillation import IGNORE_INDEX
from .stores.retrieval import RetrievalStore
from .student.lora_adapter import iter_lora_adapters


@dataclass
class EvaluationConfig:
    """Validation-pass configuration."""

    enabled: bool = False
    interval: int = 500          # step interval between val passes
    max_batches: int = 32        # cap per-domain batches for speed
    track_forgetting: bool = True
    lora_sparsity_threshold: float = 1e-4


@dataclass
class DomainMetrics:
    """Per-domain numerics produced by one eval pass."""

    domain: str
    num_tokens: int = 0
    loss_sum: float = 0.0
    samples: int = 0
    per_bucket: dict[str, tuple[float, int]] = field(default_factory=dict)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(1, self.num_tokens)

    @property
    def perplexity(self) -> float:
        loss = self.mean_loss
        return math.exp(min(20.0, loss))  # cap to prevent overflow


class ForgettingTracker:
    """Stores the first-seen per-domain baseline and computes delta."""

    def __init__(self) -> None:
        self._baseline: dict[str, float] = {}

    def observe(self, domain: str, mean_loss: float) -> Optional[float]:
        """Record ``mean_loss``; return ``loss - baseline`` once baseline set."""
        base = self._baseline.get(domain)
        if base is None:
            self._baseline[domain] = mean_loss
            return None
        return mean_loss - base

    def state_dict(self) -> dict:
        return {"baseline": dict(self._baseline)}

    def load_state_dict(self, state: dict) -> None:
        self._baseline = dict(state.get("baseline", {}))


@torch.no_grad()
def _eval_one_domain(
    student: nn.Module,
    dataloader,
    domain: str,
    device: torch.device,
    max_batches: int,
) -> DomainMetrics:
    student.eval()
    m = DomainMetrics(domain=domain)
    bucket_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])

    for step, raw in enumerate(dataloader):
        if step >= max_batches:
            break

        if not isinstance(raw, dict):
            raise RuntimeError(
                "Evaluation requires HF-style dict batches (input_ids/labels/metadata)."
            )

        input_ids = raw["input_ids"].to(device)
        labels = raw["labels"].to(device)
        meta = raw.get("metadata") or {}
        bucket_ids = meta.get("bucket_id", [""] * input_ids.shape[0])

        logits = student(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(shift_labels.shape)

        valid = (shift_labels != IGNORE_INDEX).float()
        sample_loss = (per_tok * valid).sum(dim=1)       # [B]
        sample_ntok = valid.sum(dim=1)                    # [B]
        m.loss_sum += float(sample_loss.sum().item())
        m.num_tokens += int(sample_ntok.sum().item())
        m.samples += int(input_ids.shape[0])

        for b_idx in range(input_ids.shape[0]):
            bid = bucket_ids[b_idx] if b_idx < len(bucket_ids) else ""
            if not bid:
                continue
            entry = bucket_sums[bid]
            entry[0] += float(sample_loss[b_idx].item())
            entry[1] += int(sample_ntok[b_idx].item())

    m.per_bucket = {
        bid: (loss / max(1, ntok), ntok) for bid, (loss, ntok) in bucket_sums.items()
    }
    return m


def _lora_sparsity(student: nn.Module, threshold: float) -> dict[str, float]:
    """Fraction of near-zero LoRA entries, per module name."""
    stats: dict[str, float] = {}
    for name, adapter in iter_lora_adapters(student):
        a = adapter.lora_a.detach()
        b = adapter.lora_b.detach()
        total = a.numel() + b.numel()
        zero = (a.abs() < threshold).sum().item() + (b.abs() < threshold).sum().item()
        stats[name] = float(zero) / max(1, total)
    return stats


def _retrieval_hit_rate(
    r_store: RetrievalStore,
    probe_embeddings: Optional[torch.Tensor],
    threshold: float = 0.85,
) -> float:
    """Average hits-per-probe across ``probe_embeddings`` rows."""
    if probe_embeddings is None or r_store.size == 0 or probe_embeddings.numel() == 0:
        return 0.0
    hits = 0
    for row in probe_embeddings:
        hits += r_store.hit_count(row, threshold=threshold)
    denom = max(1, probe_embeddings.shape[0] * r_store.size)
    return hits / denom


def run_evaluation(
    *,
    student: nn.Module,
    val_loaders: dict[str, Any],
    device: torch.device,
    config: EvaluationConfig,
    forgetting: Optional[ForgettingTracker] = None,
    r_store: Optional[RetrievalStore] = None,
    probe_embeddings: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Run one validation pass and return a flat metrics dict.

    Keys follow the pattern ``eval/<domain>/<metric>`` so they can be
    dropped into the existing JSON / WandB logger unchanged.
    """
    out: dict[str, Any] = {}
    per_domain: dict[str, DomainMetrics] = {}

    for domain, loader in val_loaders.items():
        dm = _eval_one_domain(student, loader, domain, device, config.max_batches)
        per_domain[domain] = dm
        out[f"eval/{domain}/loss"] = dm.mean_loss
        out[f"eval/{domain}/ppl"] = dm.perplexity
        out[f"eval/{domain}/samples"] = dm.samples
        out[f"eval/{domain}/tokens"] = dm.num_tokens
        if forgetting is not None and config.track_forgetting:
            delta = forgetting.observe(domain, dm.mean_loss)
            if delta is not None:
                out[f"eval/{domain}/forgetting_delta"] = delta
        # Per-bucket (compact — pick top-5 worst buckets by loss)
        top = sorted(dm.per_bucket.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
        for bid, (loss, ntok) in top:
            out[f"eval/{domain}/bucket/{bid}/loss"] = loss
            out[f"eval/{domain}/bucket/{bid}/tokens"] = ntok

    sparsity = _lora_sparsity(student, config.lora_sparsity_threshold)
    if sparsity:
        out["eval/lora/sparsity_mean"] = sum(sparsity.values()) / len(sparsity)
        out["eval/lora/sparsity_max"] = max(sparsity.values())

    if r_store is not None:
        out["eval/retrieval/buffer_size"] = r_store.size
        out["eval/retrieval/hit_rate"] = _retrieval_hit_rate(r_store, probe_embeddings)

    # Aggregate across domains (macro-average)
    if per_domain:
        losses = [dm.mean_loss for dm in per_domain.values() if dm.num_tokens > 0]
        if losses:
            out["eval/macro_loss"] = sum(losses) / len(losses)
            out["eval/macro_ppl"] = math.exp(min(20.0, out["eval/macro_loss"]))

    student.train()
    return out

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
    at_phase_boundaries: bool = True   # eval at the last step of each stream phase
    on_consolidation: bool = False     # eval right after any F→P merge
    # Gold-label exact-match probes (generation; phase boundaries only).
    exact_match_enabled: bool = False
    exact_match_domains: list[str] = field(default_factory=lambda: ["math"])
    exact_match_samples: int = 64
    exact_match_max_new_tokens: int = 64


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


class PhaseEvalTracker:
    """Per-(domain, phase-boundary) eval history and retention deltas.

    Retention delta for domain ``d`` = current eval loss on ``d`` minus the
    loss measured at the end of the phase that *trained* ``d`` (its
    "own-phase" loss). Positive = the model got worse on ``d`` after moving
    on — the phase-resolved forgetting number the paper reports.
    """

    def __init__(self) -> None:
        self._own_phase_loss: dict[str, float] = {}
        self._own_phase_em: dict[str, float] = {}
        self._history: list[dict[str, Any]] = []

    def record(
        self,
        *,
        phase: str,
        step: int,
        phase_domains: list[str],
        domain_loss: dict[str, float],
        domain_em: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Record a phase-boundary eval; returns per-domain retention deltas.

        ``phase_domains`` are the domains trained during the phase that just
        ended — their own-phase baselines are refreshed AFTER deltas are
        computed against the previous baselines.
        """
        domain_em = domain_em or {}
        deltas: dict[str, float] = {}
        for d, loss in domain_loss.items():
            if d in self._own_phase_loss and d not in phase_domains:
                deltas[d] = loss - self._own_phase_loss[d]
        for d in phase_domains:
            if d in domain_loss:
                self._own_phase_loss[d] = domain_loss[d]
            if d in domain_em:
                self._own_phase_em[d] = domain_em[d]
        self._history.append(
            {
                "phase": phase,
                "step": int(step),
                "loss": dict(domain_loss),
                "em": dict(domain_em),
                "retention_delta": dict(deltas),
            }
        )
        return deltas

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def retention_table(self) -> dict[str, Any]:
        """Summary for run_summary.json / paper tables."""
        return {
            "own_phase_loss": dict(self._own_phase_loss),
            "own_phase_em": dict(self._own_phase_em),
            "history": self.history,
        }

    def state_dict(self) -> dict:
        return {
            "own_phase_loss": dict(self._own_phase_loss),
            "own_phase_em": dict(self._own_phase_em),
            "history": list(self._history),
        }

    def load_state_dict(self, state: dict) -> None:
        self._own_phase_loss = dict(state.get("own_phase_loss", {}))
        self._own_phase_em = dict(state.get("own_phase_em", {}))
        self._history = list(state.get("history", []))


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


# -- gold-label exact match (generation probes) ------------------------------


@dataclass
class GoldProbeSet:
    """A fixed, seed-deterministic set of gold-labeled rows for one domain."""

    domain: str
    rows: list[dict[str, Any]]


def build_gold_probes(
    dataset: Any,  # HivemindHFDataset (needs gold_indices / __getitem__)
    domains: list[str],
    num_samples: int,
    seed: int,
) -> dict[str, GoldProbeSet]:
    """Pick a fixed probe set per domain from rows with ``has_gold_label``."""
    import numpy as np

    probes: dict[str, GoldProbeSet] = {}
    for domain in domains:
        try:
            gold = list(dataset.gold_indices(domain))
        except (KeyError, AttributeError):
            continue
        if not gold:
            continue
        rng = np.random.default_rng(seed + len(domain))
        if len(gold) > num_samples:
            gold = [int(i) for i in rng.choice(gold, size=num_samples, replace=False)]
        rows = [dataset[i] for i in gold]
        rows = [r for r in rows if r.get("target_text")]
        if rows:
            probes[domain] = GoldProbeSet(domain=domain, rows=rows)
    return probes


def normalize_answer(text: str) -> str:
    """Normalize a model/gold answer for exact-match comparison.

    Prefers ``\\boxed{...}`` content, then the last number in the text,
    else the lowercased stripped string.
    """
    import re

    text = (text or "").strip()
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        num = numbers[-1]
        return num.rstrip("0").rstrip(".") if "." in num else num
    return " ".join(text.lower().split())


@torch.no_grad()
def exact_match_eval(
    student: nn.Module,
    tokenizer: Any,
    probes: GoldProbeSet,
    device: torch.device,
    max_new_tokens: int = 64,
    max_prompt_tokens: int = 512,
) -> float:
    """Greedy-decode each probe prompt and exact-match against target_text.

    Uses the plain ``student(tokens) -> logits`` interface so it works for
    every backbone (in-repo, HF, unsloth). One prompt at a time — probes run
    only at phase boundaries, so throughput is acceptable.
    """
    student.eval()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    correct = 0

    for row in probes.rows:
        prompt_ids = tokenizer(
            row["input_text"],
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        )["input_ids"].to(device)

        generated: list[int] = []
        tokens = prompt_ids
        for _ in range(max_new_tokens):
            logits = student(tokens)
            next_id = int(logits[0, -1].argmax().item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], device=device)], dim=1
            )

        answer = tokenizer.decode(generated, skip_special_tokens=True)
        if normalize_answer(answer) == normalize_answer(row["target_text"]):
            correct += 1

    student.train()
    return correct / max(1, len(probes.rows))

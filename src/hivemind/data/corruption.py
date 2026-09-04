"""Corrupted-teacher stream (task T4, experiment E5).

One teacher is made bad for one recurrent phase in a controlled,
deterministic, digest-tracked way, so the paper can ask whether a bad
teacher's influence stays out of permanent memory and can be reverted.

Two pieces:

* :class:`CorruptionSpec` — the ``data.corrupt_teacher`` config block.
* :class:`CorruptedTeacherView` — a thin ``Dataset`` view that rewrites the
  teacher fields of the rows the schedule selected. Applied by global row
  index, so it is independent of batch size and of how DataLoader threads
  rows to the collator; the schedule (built on the raw dataset) and every
  eval loader stay clean by construction.

Selection lives in :class:`~hivemind.data.stream.StreamSchedule` because
only the schedule knows which rows are served in which phase. It is
**batch-coherent**: whole bucket subsets and whole background/novel steps
are corrupted, never individual rows inside a batch. With ``batch_size=2``
and bucket rows served ~19× each, per-row selection would make every
bucket's batches alternate clean/corrupt across visits (and mix within one
batch), so the representative-row teacher label would be wrong half the
time and merge attribution a coin flip.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from torch.utils.data import Dataset

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WRONG_SUFFIX = " Final answer: 0"


@dataclass
class CorruptionSpec:
    """``data.corrupt_teacher`` — defaults reproduce the clean stream."""

    enabled: bool = False
    domain: str = "math"
    phase: str = "math_recurrent"      # only rows served during this phase
    fraction: float = 0.5              # of that phase's steps (batch-coherent)
    mode: str = "shuffle"              # shuffle | degrade
    # Confidence written on corrupted rows. ``None`` keeps each row's own
    # confidence so E5 measures the ROUTING, not the confidence gate; set
    # 0.2 for the low-confidence arm.
    confidence: Optional[float] = None
    tag: str = "math_teacher_corrupted"

    def __post_init__(self) -> None:
        if self.mode not in ("shuffle", "degrade"):
            raise ValueError(
                f"corrupt_teacher.mode={self.mode!r}; allowed: 'shuffle' | 'degrade'"
            )
        if not 0.0 <= float(self.fraction) <= 1.0:
            raise ValueError("corrupt_teacher.fraction must be in [0, 1]")
        self.fraction = float(self.fraction)
        if self.confidence is not None:
            self.confidence = float(self.confidence)

    def digest_payload(self) -> dict[str, Any]:
        """What goes into the stream digest (only when enabled)."""
        payload = asdict(self)
        payload["selection"] = "bucket+batch"
        return payload


def parse_corruption_config(raw: Optional[dict[str, Any]]) -> CorruptionSpec:
    if not raw:
        return CorruptionSpec()
    return CorruptionSpec(**{k: v for k, v in dict(raw).items()})


def degrade_text(text: str, keep_fraction: float = 0.25) -> str:
    """Truncate to the first ``keep_fraction`` of whitespace tokens and append
    a wrong final answer: the last number in the ORIGINAL text plus one, or a
    fixed wrong suffix when there is no number."""
    tokens = str(text).split()
    keep = max(1, int(len(tokens) * keep_fraction)) if tokens else 0
    head = " ".join(tokens[:keep])
    numbers = _NUMBER_RE.findall(str(text))
    if numbers:
        last = numbers[-1]
        try:
            if "." in last:
                wrong = f"{float(last) + 1:g}"
            else:
                wrong = str(int(last) + 1)
        except ValueError:  # pragma: no cover - regex guarantees a number
            wrong = "0"
        return f"{head} Final answer: {wrong}"
    return head + _WRONG_SUFFIX


@dataclass
class CorruptionPlan:
    """Everything the view needs: which rows, and the shuffle partners."""

    spec: CorruptionSpec
    corrupted: set[int]
    partner_of: dict[int, int]  # shuffle mode: row → row whose teacher text it takes

    @property
    def size(self) -> int:
        return len(self.corrupted)


class CorruptedTeacherView(Dataset):
    """Row-level view applying a :class:`CorruptionPlan` on top of a dataset.

    Corrupted rows get: ``teacher_output_text`` replaced (partner's text
    for ``shuffle``, :func:`degrade_text` for ``degrade``), ``teacher_id``
    set to the tag (so decisions and R entries are attributed to a distinct
    teacher), ``teacher_confidence`` overridden when the spec says so,
    ``teacher_logits_path`` cleared (the original teacher's cached logits
    would no longer align with the response), and ``corrupted = True``.
    Everything else is delegated to the underlying dataset, including the
    stream accessors the schedule and preflight use.
    """

    def __init__(self, dataset: Any, plan: CorruptionPlan) -> None:
        self.dataset = dataset
        self.plan = plan

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:  # delegation for accessors
        if name in ("dataset", "plan"):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataset[idx]
        if idx < 0:
            idx += len(self.dataset)
        if idx not in self.plan.corrupted:
            return row
        row = dict(row)
        spec = self.plan.spec
        original = str(row.get("teacher_output_text") or "")
        if spec.mode == "shuffle" and idx in self.plan.partner_of:
            partner = self.dataset[self.plan.partner_of[idx]]
            row["teacher_output_text"] = str(partner.get("teacher_output_text") or "")
        else:
            row["teacher_output_text"] = degrade_text(original)
        row["teacher_id"] = spec.tag
        if spec.confidence is not None:
            row["teacher_confidence"] = spec.confidence
        row["teacher_logits_path"] = None
        row["corrupted"] = True
        return row

"""HuggingFace-backed multi-domain dataset.

Two loading paths per domain, selected by :class:`DomainSpec`:

* **Layer C (preferred)** — a training-ready pre-joined view
  (``{org}/{domain}-layerC-*``) with canonical sample fields AND teacher
  supervision in one table, plus proper ``train``/``validation``/``test``
  splits. The Arrow-backed ``datasets.Dataset`` handle is kept as-is
  (memory-mapped) — 160k-row tables are NOT materialized into python lists.
* **Layer A + B join (legacy)** — two repos joined on ``sample_id``
  (``-layerA-final`` × ``-layerB-final``), materialized into dicts. This is
  the exact behavior older configs relied on; a plain string domain spec
  (``domains: [code, math]``) still resolves to it.

Rows are returned as plain python dicts. Tokenization happens in the
``DistillCollator`` so the dataset can be cached / inspected cheaply.

The dataset also exposes index accessors used by the stream scheduler
(:mod:`hivemind.data.stream`): ``indices_for_domain``, ``indices_by_bucket``,
``gold_indices``.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DomainSpec:
    """Where one domain's data lives on the HF hub.

    ``layer_c`` names a pre-joined training-ready repo; when absent (or when
    it fails to load), ``layer_a`` × ``layer_b`` are joined on ``sample_id``.
    ``max_rows`` caps this domain individually, overriding the global
    ``max_rows_per_domain``.
    """

    name: str
    layer_c: Optional[str] = None
    layer_a: Optional[str] = None
    layer_b: Optional[str] = None
    max_rows: Optional[int] = None


def normalize_domain_specs(
    raw: Sequence["str | dict[str, Any] | DomainSpec"],
    repo_prefix: str = "hivemind-research",
) -> list[DomainSpec]:
    """Normalize config-provided domain entries into :class:`DomainSpec`.

    A plain string (``"code"``) reproduces the legacy behavior exactly:
    join ``{prefix}/code-layerA-final`` × ``{prefix}/code-layerB-final``.
    A dict maps onto DomainSpec fields (unknown keys raise).
    """
    specs: list[DomainSpec] = []
    for entry in raw:
        if isinstance(entry, DomainSpec):
            specs.append(entry)
        elif isinstance(entry, str):
            specs.append(
                DomainSpec(
                    name=entry,
                    layer_a=f"{repo_prefix}/{entry}-layerA-final",
                    layer_b=f"{repo_prefix}/{entry}-layerB-final",
                )
            )
        elif isinstance(entry, dict):
            unknown = set(entry) - {f.name for f in DomainSpec.__dataclass_fields__.values()}
            if unknown:
                raise ValueError(
                    f"Unknown DomainSpec keys {sorted(unknown)} in {entry!r}; "
                    f"allowed: name, layer_c, layer_a, layer_b, max_rows"
                )
            if "name" not in entry:
                raise ValueError(f"Domain spec dict requires 'name': {entry!r}")
            specs.append(DomainSpec(**entry))
        else:
            raise TypeError(f"Unsupported domain spec: {entry!r}")
    return specs


def domain_names(domains: Sequence["str | dict[str, Any] | DomainSpec"]) -> list[str]:
    """Just the names, for callers that iterate domains for eval/metrics."""
    out = []
    for entry in domains:
        if isinstance(entry, DomainSpec):
            out.append(entry.name)
        elif isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            out.append(str(entry["name"]))
        else:
            raise TypeError(f"Unsupported domain spec: {entry!r}")
    return out


@dataclass
class HivemindHFConfig:
    """Configuration for the HF-backed dataset.

    ``domains`` entries may be plain strings (legacy A+B join) or dicts /
    :class:`DomainSpec` (explicit repos, Layer C preferred).

    ``teacher_logits_root`` may point at a local directory whose layout
    mirrors the ``teacher_logits_path`` strings stored in Layer B/C. Leaving
    it ``None`` disables logit-KD — the collator returns
    ``teacher_logits_mask`` all zeros and the objective falls back to CE.
    """

    domains: list[Any] = field(default_factory=lambda: ["code"])
    repo_prefix: str = "hivemind-research"
    split: str = "train"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    max_seq_len: int = 2048
    teacher_logits_root: Optional[str] = None
    cache_dir: Optional[str] = None
    streaming: bool = False
    max_rows_per_domain: Optional[int] = None


_LAYER_A_REQUIRED = {"sample_id", "input_text"}
_LAYER_B_REQUIRED = {"sample_id", "teacher_id", "teacher_output_text"}
_LAYER_C_REQUIRED = {"sample_id", "input_text", "teacher_id", "teacher_output_text"}


class _Segment:
    """One domain's backing store: an Arrow dataset or a list of dicts."""

    def __init__(self, name: str, backing: Any, length: int, arrow: bool) -> None:
        self.name = name
        self.backing = backing
        self.length = length
        self.arrow = arrow

    def row(self, local_idx: int) -> dict[str, Any]:
        row = dict(self.backing[local_idx])
        row.setdefault("domain", self.name)
        return row

    def column(self, key: str) -> list[Any]:
        if self.arrow:
            if key not in self.backing.column_names:
                return [None] * self.length
            return list(self.backing[key])
        return [r.get(key) for r in self.backing]


class HivemindHFDataset(Dataset):
    """Multi-domain training-ready dataset (Layer C direct or A×B joined).

    Each element is a dict; the collator only requires:

    * ``sample_id`` (str)
    * ``input_text`` (str)
    * ``teacher_id`` (str)
    * ``teacher_output_text`` (str, optional)
    * ``teacher_logits_path`` (str, optional)
    * ``target_text`` (str, optional — gold CE supervision)
    * ``bucket_id`` (str, optional — controller repetition key)
    * ``domain`` (str — backfilled from the spec when missing)
    """

    def __init__(self, cfg: HivemindHFConfig) -> None:
        self.cfg = cfg
        self.specs = normalize_domain_specs(cfg.domains, cfg.repo_prefix)
        self._segments: list[_Segment] = []
        self._starts: list[int] = []  # cumulative start index per segment
        self._bucket_cache: dict[str, dict[str, list[int]]] = {}

        for spec in self.specs:
            self._load_domain(spec)

        total = 0
        self._starts = []
        for seg in self._segments:
            self._starts.append(total)
            total += seg.length
        self._total = total

        if total == 0:
            raise RuntimeError(
                "HivemindHFDataset: no usable rows for domains="
                f"{[s.name for s in self.specs]} split={cfg.split}. "
                "Check the repos (see docs/dataset_preperation.md)."
            )

        logger.info(
            "HivemindHFDataset loaded: %d rows across domains=%s split=%s",
            total,
            [s.name for s in self.specs],
            cfg.split,
        )

    # -- loading -----------------------------------------------------------

    def _cap(self, spec: DomainSpec) -> Optional[int]:
        if spec.max_rows is not None:
            return int(spec.max_rows)
        if self.cfg.max_rows_per_domain is not None:
            return int(self.cfg.max_rows_per_domain)
        return None

    def _load_domain(self, spec: DomainSpec) -> None:
        if spec.layer_c:
            ds = self._try_load(spec.layer_c)
            if ds is not None:
                self._check_schema(ds, _LAYER_C_REQUIRED, spec.layer_c)
                cap = self._cap(spec)
                if cap is not None and hasattr(ds, "select") and len(ds) > cap:
                    ds = ds.select(range(cap))
                self._segments.append(_Segment(spec.name, ds, len(ds), arrow=True))
                logger.info(
                    "Domain %s: %d rows from Layer C %s (%s)",
                    spec.name, len(ds), spec.layer_c, self.cfg.split,
                )
                return
            logger.warning(
                "Domain %s: Layer C %s unavailable; falling back to A+B join.",
                spec.name, spec.layer_c,
            )
        self._load_domain_join(spec)

    def _load_domain_join(self, spec: DomainSpec) -> None:
        repo_a = spec.layer_a or f"{self.cfg.repo_prefix}/{spec.name}-layerA-final"
        repo_b = spec.layer_b or f"{self.cfg.repo_prefix}/{spec.name}-layerB-final"

        layer_b = self._try_load(repo_b)
        if layer_b is None:
            logger.warning("Domain %s: Layer B missing at %s; skipping.", spec.name, repo_b)
            return
        self._check_schema(layer_b, _LAYER_B_REQUIRED, repo_b)
        layer_b_by_id = {row["sample_id"]: row for row in layer_b}

        layer_a = self._try_load(repo_a)
        if layer_a is None:
            logger.warning(
                "Domain %s: Layer A missing at %s; rows without input_text will be "
                "skipped until it is published.",
                spec.name, repo_a,
            )
            return
        self._check_schema(layer_a, _LAYER_A_REQUIRED, repo_a)

        cap = self._cap(spec)
        rows: list[dict[str, Any]] = []
        for row_a in layer_a:
            sid = row_a.get("sample_id")
            if sid is None:
                continue
            row_b = layer_b_by_id.get(sid)
            if row_b is None:
                continue
            merged: dict[str, Any] = dict(row_a)
            for k, v in row_b.items():
                merged.setdefault(k, v)
            merged.setdefault("domain", spec.name)
            rows.append(merged)
            if cap is not None and len(rows) >= cap:
                break

        if rows:
            self._segments.append(_Segment(spec.name, rows, len(rows), arrow=False))
        logger.info(
            "Domain %s: %d rows joined (A=%d, B=%d)",
            spec.name, len(rows), len(layer_a), len(layer_b),
        )

    def _try_load(self, repo: str) -> Optional[Any]:
        from datasets import load_dataset

        try:
            return load_dataset(
                repo,
                split=self.cfg.split,
                cache_dir=self.cfg.cache_dir,
                streaming=self.cfg.streaming,
            )
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 — HF raises many subclasses we want to tolerate
            logger.warning("load_dataset(%s) failed: %s", repo, exc)
            return None

    def _check_schema(self, ds: Any, required: set[str], repo: str) -> None:
        cols: Optional[list[str]] = None
        if hasattr(ds, "column_names") and ds.column_names is not None:
            cols = list(ds.column_names)
        if cols is not None:
            missing = required - set(cols)
            if missing:
                raise ValueError(
                    f"{repo}: missing required columns {sorted(missing)}. "
                    f"Got {sorted(cols)}."
                )

    # -- Dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self._total
        if not 0 <= idx < self._total:
            raise IndexError(idx)
        seg_i = bisect_right(self._starts, idx) - 1
        return self._segments[seg_i].row(idx - self._starts[seg_i])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(self._total):
            yield self[i]

    @property
    def domains(self) -> list[str]:
        return [s.name for s in self.specs]

    # -- stream-scheduler accessors ---------------------------------------

    def _segment_for(self, name: str) -> tuple[_Segment, int]:
        for seg, start in zip(self._segments, self._starts):
            if seg.name == name:
                return seg, start
        raise KeyError(f"Domain '{name}' not loaded (have {self.domains})")

    def indices_for_domain(self, name: str) -> range:
        seg, start = self._segment_for(name)
        return range(start, start + seg.length)

    def indices_by_bucket(self, name: str) -> dict[str, list[int]]:
        """Global indices grouped by ``bucket_id``, lazily built and cached."""
        if name not in self._bucket_cache:
            seg, start = self._segment_for(name)
            buckets: dict[str, list[int]] = {}
            for local_i, bucket in enumerate(seg.column("bucket_id")):
                key = str(bucket) if bucket else f"{name}__nobucket"
                buckets.setdefault(key, []).append(start + local_i)
            self._bucket_cache[name] = buckets
        return self._bucket_cache[name]

    def gold_indices(self, name: str) -> list[int]:
        """Global indices of rows with a trusted gold label."""
        seg, start = self._segment_for(name)
        return [
            start + i
            for i, flag in enumerate(seg.column("has_gold_label"))
            if bool(flag)
        ]

"""HuggingFace-backed multi-domain dataset.

Loads two HF repos per domain and joins on ``sample_id``:

* ``{repo_prefix}/{domain}-layerA-final`` — canonical samples
  (``input_text``, ``target_text``, ``domain``, ``subdomain``,
  ``difficulty``, ``bucket_id``, ``has_gold_label``, ...)
* ``{repo_prefix}/{domain}-layerB-final`` — teacher supervision
  (``teacher_id``, ``teacher_output_text``, ``teacher_logits_path``,
  ``teacher_confidence``, ``teacher_entropy``, ...)

When Layer A is not yet published for a domain (currently: math), rows are
dropped until an ``input_text`` source is available — the collator cannot
tokenize a missing input. This is a soft skip with a visible log line,
not a raise, so multi-domain training isn't blocked by one gap.

Rows are returned as plain python dicts. Tokenization happens in the
``DistillCollator`` so the dataset can be cached / inspected cheaply.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class HivemindHFConfig:
    """Configuration for the HF-backed dataset.

    ``teacher_logits_root`` may point at a local directory whose layout
    mirrors the ``teacher_logits_path`` strings stored in Layer B. Leaving
    it ``None`` disables KD — the collator returns ``teacher_logits_mask``
    with all zeros and the objective falls back to CE only.
    """

    domains: list[str] = field(default_factory=lambda: ["code"])
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


class HivemindHFDataset(Dataset):
    """Multi-domain Layer A × Layer B joined view.

    Each element is a dict with at least the union of Layer A + Layer B
    fields plus ``domain``. The specific keys emitted depend on what's
    present in the upstream repos, but the collator only requires:

    * ``sample_id`` (str)
    * ``input_text`` (str)
    * ``teacher_id`` (str)
    * ``teacher_output_text`` (str, optional)
    * ``teacher_logits_path`` (str, optional)
    * ``target_text`` (str, optional — gold CE supervision)
    * ``bucket_id`` (str, optional — controller repetition key)
    * ``domain`` (str — backfilled from config when missing)
    """

    def __init__(self, cfg: HivemindHFConfig) -> None:
        self.cfg = cfg
        self._rows: list[dict[str, Any]] = []
        self._bucket_index: dict[str, int] = {}

        for domain in cfg.domains:
            self._load_domain(domain)

        if not self._rows:
            raise RuntimeError(
                f"HivemindHFDataset: no usable rows after joining domains={cfg.domains}. "
                "Check that Layer A is published for each domain (see docs/dataset_preperation.md)."
            )

        logger.info(
            "HivemindHFDataset loaded: %d rows across domains=%s split=%s",
            len(self._rows),
            cfg.domains,
            cfg.split,
        )

    # -- loading -----------------------------------------------------------

    def _load_domain(self, domain: str) -> None:
        repo_a = f"{self.cfg.repo_prefix}/{domain}-layerA-final"
        repo_b = f"{self.cfg.repo_prefix}/{domain}-layerB-final"

        layer_b = self._try_load(repo_b, domain)
        if layer_b is None:
            logger.warning("Domain %s: Layer B missing at %s; skipping.", domain, repo_b)
            return
        self._check_schema(layer_b, _LAYER_B_REQUIRED, repo_b)
        layer_b_by_id = {row["sample_id"]: row for row in layer_b}

        layer_a = self._try_load(repo_a, domain)
        if layer_a is None:
            logger.warning(
                "Domain %s: Layer A missing at %s; rows without input_text will be skipped "
                "until it is published.",
                domain,
                repo_a,
            )
            return
        self._check_schema(layer_a, _LAYER_A_REQUIRED, repo_a)

        appended = 0
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
            merged.setdefault("domain", domain)
            self._rows.append(merged)
            appended += 1
            if self.cfg.max_rows_per_domain and appended >= self.cfg.max_rows_per_domain:
                break

        logger.info(
            "Domain %s: %d rows joined (A=%d, B=%d)",
            domain,
            appended,
            len(layer_a),
            len(layer_b),
        )

    def _try_load(self, repo: str, domain: str) -> Optional[Any]:
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
        # `datasets` exposes feature names via .column_names for non-streaming;
        # for streaming we peek at the first row and trust it.
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
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    @property
    def domains(self) -> list[str]:
        return list(self.cfg.domains)

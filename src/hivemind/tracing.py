"""Structured event trace + per-module ledger.

This is the paper's audit trail of *when and why parameters change*: every
routing decision, consolidation, and phase transition is appended to a JSONL
event file, and cumulative per-module counters are kept in a checkpointable
ledger.

Event records are one JSON object per line, discriminated by ``type``:

- ``run_config``  — once per (re)start: policy thresholds, Top-M/Top-K.
- ``decision``    — one per Top-M module per step: action R/F/P, all signal
                    values (S, C, V, R + components, stability_adam,
                    grad_norm), coords opened, bucket id.
- ``consolidation`` — a completed F→P merge: module, strategy, pre-merge
                    signals, forced flag.
- ``router``      — periodic router health: metadata-miss count.
- ``resume``      — training resumed from a checkpoint (makes appended
                    traces self-describing).
- ``phase_start`` / ``phase_end`` — stream phase boundaries (experiment
                    harness).

The file is opened in append mode so traces survive resume; a ``resume``
event marks each restart point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EventTrace:
    """Buffered append-mode JSONL event writer."""

    def __init__(
        self,
        path: str | Path = "logs/events.jsonl",
        flush_every: int = 50,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self.flush_every = max(1, flush_every)
        self._buffer: list[str] = []
        self._fh = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        """Append one event (caller supplies the ``type`` field)."""
        if not self.enabled:
            return
        self._buffer.append(json.dumps(event, default=_json_default))
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self._fh is None or not self._buffer:
            return
        self._fh.write("\n".join(self._buffer) + "\n")
        self._fh.flush()
        self._buffer = []

    def close(self) -> None:
        if self._fh is not None:
            self.flush()
            self._fh.close()
            self._fh = None


def _json_default(obj: Any) -> Any:
    """Best-effort serialization for tensors/numpy scalars in events."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _new_entry() -> dict[str, int]:
    """Fresh per-module counters (a factory, never a shared template)."""
    return {
        "r_count": 0,
        "f_count": 0,
        "p_count": 0,
        "times_opened": 0,
        "last_opened_step": -1,
        "total_coords_opened": 0,
        "consolidations": 0,
        "last_consolidated_step": -1,
        # Permanent-write accounting (task T7, the budget axis of Figure 1):
        # coordinates opened by P actions on the block's BASE weights, and
        # base coordinates rewritten by LoRA→base merges of this block.
        "p_coords_opened": 0,
        "merged_coords": 0,
        # F writes that came from an R-store replay (T2); they are also
        # counted in f_count, so subtract to get policy-only F writes.
        "replay_count": 0,
    }


def _new_global() -> dict[str, int]:
    """Run-level counters that no single module owns."""
    return {
        # Steps that trained (skipped steps excluded).
        "steps": 0,
        # Σ over steps of coordinates open that step (controller-indexed
        # params only), whatever opened them: masks, or — with masking off —
        # every trainable indexed coordinate. Divided by steps × indexed
        # coordinates this is the mean active fraction.
        "coords_opened": 0,
        # Σ over steps of base coordinates that updated WITHOUT a mask
        # (controller off / optim.masked=false): the permanent-write count
        # of a full-FT run, which has no P actions to count.
        "unmasked_base_coords": 0,
    }


class ModuleLedger:
    """Cumulative per-module write counters, checkpointed as ``ledger.pt``.

    Answers "how often has L3.ffn.F been opened, when last, how many
    coordinates total" without replaying the whole event trace.
    """

    _EMPTY = _new_entry()  # kept for callers that read the field list

    def __init__(self) -> None:
        self._modules: dict[str, dict[str, int]] = {}
        self._global: dict[str, int] = _new_global()

    def _entry(self, module: str) -> dict[str, int]:
        if module not in self._modules:
            self._modules[module] = _new_entry()
        return self._modules[module]

    def record_action(
        self,
        module: str,
        store: str,
        step: int,
        coords_opened: int = 0,
        *,
        replay: bool = False,
    ) -> None:
        entry = self._entry(module)
        key = f"{store.lower()}_count"
        if key in entry:
            entry[key] += 1
        if replay:
            entry["replay_count"] += 1
        if coords_opened > 0:
            entry["times_opened"] += 1
            entry["last_opened_step"] = int(step)
            entry["total_coords_opened"] += int(coords_opened)
            if store == "P" and module.endswith(".P"):
                entry["p_coords_opened"] += int(coords_opened)

    def record_consolidation(
        self, module: str, step: int, *, merged_coords: int = 0
    ) -> None:
        entry = self._entry(module)
        entry["consolidations"] += 1
        entry["last_consolidated_step"] = int(step)
        entry["merged_coords"] += int(merged_coords)

    def record_step(self, coords_opened: int, unmasked_base_coords: int = 0) -> None:
        """Close one trained step (run-level active-fraction / budget sums)."""
        self._global["steps"] += 1
        self._global["coords_opened"] += int(coords_opened)
        self._global["unmasked_base_coords"] += int(unmasked_base_coords)

    def summary(self) -> dict[str, dict[str, int]]:
        """Per-module counters, sorted by module name."""
        return {k: dict(v) for k, v in sorted(self._modules.items())}

    def totals(self) -> dict[str, int]:
        out = {
            "r_count": 0,
            "f_count": 0,
            "p_count": 0,
            "consolidations": 0,
            "p_coords_opened": 0,
            "merged_coords": 0,
            "replay_count": 0,
        }
        for entry in self._modules.values():
            for key in out:
                out[key] += entry.get(key, 0)
        out.update(self._global)
        return out

    # -- serialisation ----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "modules": {k: dict(v) for k, v in self._modules.items()},
            "global": dict(self._global),
        }

    def load_state_dict(self, state: dict) -> None:
        self._modules = {
            k: {**_new_entry(), **v} for k, v in state.get("modules", {}).items()
        }
        self._global = {**_new_global(), **state.get("global", {})}

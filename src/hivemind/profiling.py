"""Memory + wall-clock profiling for training runs.

Answers the paper's "report memory and training time" requirement:

* Per-section step timing (data / teacher / forward / backward / signals /
  policy+writes / optimizer / consolidation / eval) as EMAs, so the
  controller's overhead is separable from the model math.
* Per-phase peak CUDA memory (reset at each stream phase boundary).
* Tokens/sec throughput.
* A final ``run_summary.json`` with totals — written on normal exit AND on
  SIGINT emergency save (``interrupted: true``).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

CONTROLLER_SECTIONS = ("signals", "policy_write", "consolidation")


class StepTimer:
    """Named-section wall-clock EMAs (perf_counter based)."""

    def __init__(self, ema_beta: float = 0.98, sync_cuda: bool = False) -> None:
        self.ema_beta = float(ema_beta)
        self.sync_cuda = bool(sync_cuda)
        self._ema: dict[str, float] = {}
        self._totals: dict[str, float] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prev = self._ema.get(name)
            self._ema[name] = (
                elapsed
                if prev is None
                else self.ema_beta * prev + (1 - self.ema_beta) * elapsed
            )
            self._totals[name] = self._totals.get(name, 0.0) + elapsed

    def ema(self) -> dict[str, float]:
        return dict(self._ema)

    def totals(self) -> dict[str, float]:
        return dict(self._totals)


class RunProfiler:
    """Owns step timing, throughput, and per-phase peak-memory accounting."""

    def __init__(
        self,
        device: torch.device,
        enabled: bool = True,
        sync_cuda: bool = False,
        ema_beta: float = 0.98,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.timer = StepTimer(ema_beta=ema_beta, sync_cuda=sync_cuda)
        self._run_start = time.perf_counter()
        self._steps = 0
        self._tokens = 0
        self._step_ema: Optional[float] = None
        self._ema_beta = float(ema_beta)
        self._last_step_start: Optional[float] = None
        self._phase_peaks: dict[str, float] = {}
        self._cuda = device.type == "cuda" and torch.cuda.is_available()

    # -- step accounting ---------------------------------------------------

    def step_start(self) -> None:
        if self.enabled:
            self._last_step_start = time.perf_counter()

    def step_end(self, tokens_in_step: int) -> None:
        if not self.enabled or self._last_step_start is None:
            return
        elapsed = time.perf_counter() - self._last_step_start
        self._step_ema = (
            elapsed
            if self._step_ema is None
            else self._ema_beta * self._step_ema + (1 - self._ema_beta) * elapsed
        )
        self._steps += 1
        self._tokens += int(tokens_in_step)

    # -- phase memory ------------------------------------------------------

    def on_phase_start(self, phase: str) -> None:
        if self.enabled and self._cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_phase_end(self, phase: str) -> dict[str, Any]:
        if not (self.enabled and self._cuda):
            return {}
        peak_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        self._phase_peaks[phase] = peak_gb
        return {"mem/phase_peak_gb": peak_gb, "phase": phase}

    # -- metrics -----------------------------------------------------------

    def step_metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        out: dict[str, float] = {}
        if self._step_ema:
            out["time/step_ema_s"] = self._step_ema
            if self._steps > 0 and self._tokens > 0:
                out["time/tokens_per_sec"] = (
                    self._tokens / max(1e-9, time.perf_counter() - self._run_start)
                )
        section_ema = self.timer.ema()
        for name, val in section_ema.items():
            out[f"time/{name}_ema_s"] = val
        # Controller overhead as a fraction of the step.
        if self._step_ema:
            ctrl = sum(section_ema.get(s, 0.0) for s in CONTROLLER_SECTIONS)
            out["time/controller_fraction"] = ctrl / max(1e-9, self._step_ema)
        if self._cuda:
            out["mem/allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
            out["mem/peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
        return out

    def finalize(self) -> dict[str, Any]:
        wall = time.perf_counter() - self._run_start
        out: dict[str, Any] = {
            "wall_clock_s": wall,
            "steps": self._steps,
            "tokens": self._tokens,
            "tokens_per_sec": self._tokens / max(1e-9, wall),
            "section_totals_s": self.timer.totals(),
            "phase_peak_mem_gb": dict(self._phase_peaks),
        }
        totals = self.timer.totals()
        ctrl_total = sum(totals.get(s, 0.0) for s in CONTROLLER_SECTIONS)
        out["controller_time_s"] = ctrl_total
        out["controller_fraction"] = ctrl_total / max(1e-9, wall)
        if self._cuda:
            out["final_peak_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
        return out


def write_run_summary(
    run_dir: "str | Path",
    *,
    profiler: Optional[RunProfiler] = None,
    final_metrics: Optional[dict[str, Any]] = None,
    retention: Optional[dict[str, Any]] = None,
    ledger_totals: Optional[dict[str, Any]] = None,
    interrupted: bool = False,
    extra: Optional[dict[str, Any]] = None,
    permanent_writes: Optional[dict[str, Any]] = None,
    indexed_coords: Optional[int] = None,
    active_fraction_mean: Optional[float] = None,
) -> Path:
    """Write ``run_summary.json`` into ``run_dir`` (best-effort, atomic-ish).

    ``permanent_writes`` / ``indexed_coords`` / ``active_fraction_mean`` are
    the budget-axis numbers of the paper's Figure 1 (task T7); all optional
    so older callers keep working.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "interrupted": bool(interrupted),
        "timestamp": time.time(),
    }
    if profiler is not None:
        payload["profile"] = profiler.finalize()
    if final_metrics:
        payload["final_metrics"] = {
            k: v for k, v in final_metrics.items() if isinstance(v, (int, float, str))
        }
    if retention:
        payload["retention"] = retention
    if ledger_totals:
        payload["ledger_totals"] = ledger_totals
    if permanent_writes is not None:
        payload["permanent_writes"] = permanent_writes
    if indexed_coords is not None:
        payload["indexed_coords"] = int(indexed_coords)
    if active_fraction_mean is not None:
        payload["active_fraction_mean"] = float(active_fraction_mean)
    if extra:
        payload["extra"] = extra
    path = run_dir / "run_summary.json"
    tmp = run_dir / ".run_summary.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.rename(path)
    return path

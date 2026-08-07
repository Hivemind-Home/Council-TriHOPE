"""Phased continual-learning stream with controlled recurrence.

Orders training batches into named PHASES over domains so that the R/F/P
controller sees exactly the stream structure the paper's study needs:

* ``random``    — iid draws from one domain (warm-up / background).
* ``recurrent`` — a fixed subset of buckets revisited on a fixed period, so
  repetition rises and stability settles enough for P-consolidation to fire.
* ``novel``     — isolated fresh rows, each presented once with a unique
  synthetic bucket id (the "isolated novel batch" probe: should route to R).
* ``mixed``     — weighted iid draws across several domains (interleaved tail).

The full index sequence is precomputed deterministically from the seed at
construction, which makes the stream (a) reproducible, (b) resumable from any
step, and (c) cheap to validate: ``config_digest()`` is stored in checkpoint
metadata and compared on resume so a silently-edited stream config cannot
corrupt a continued run.

Batches in recurrent phases are bucket-pure (the controller keys repetition
on the batch's first sample). Novel batches get their bucket via
``bucket_override(step)`` which the training loop consults.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
from torch.utils.data import Sampler


@dataclass
class RecurrenceSpec:
    """Controls the recurrent mode of one phase.

    ``num_buckets`` buckets are chosen deterministically; each holds a fixed
    subset of ``samples_per_bucket`` rows. Bucket ``b`` occupies the stream
    slots where ``step_in_phase % revisit_period == slot_b`` (slots spread
    evenly). Slots not owned by a bucket are iid background draws;
    ``novel_fraction`` of those become fresh single-use rows instead.
    ``reuse_from`` replays the exact bucket/sample selection of an earlier
    phase (steps-to-recover measurements).

    ``exclude_chosen_buckets_from_background`` keeps this phase's background
    slots away from the buckets it chose. Without it, only the *sampled rows*
    are reserved, so a background draw can still land on another row of a
    chosen bucket — and since the controller keys repetition on the batch's
    bucket id, that re-increments the very counter the phase is trying to
    measure. With ``n/(n+k)`` at ``k=10`` the frequency term saturates above
    0.9 after ~45 such hits, so repetition would read high from the start of
    the phase and never ramp. The scope is deliberately this phase's own
    background pool; later phases (a mixed replay tail) may revisit these
    buckets on purpose.
    """

    num_buckets: int = 8
    samples_per_bucket: int = 16
    revisit_period: Optional[int] = None  # default: num_buckets (pure cycling)
    novel_fraction: float = 0.0
    reuse_from: Optional[str] = None
    exclude_chosen_buckets_from_background: bool = False


@dataclass
class PhaseSpec:
    """One contiguous segment of the stream."""

    name: str
    steps: int
    mode: str  # random | recurrent | novel | mixed
    domain: Optional[str] = None
    domains: Optional[list[str]] = None
    weights: Optional[list[float]] = None
    recurrence: Optional[RecurrenceSpec] = None

    def __post_init__(self) -> None:
        if self.mode not in ("random", "recurrent", "novel", "mixed"):
            raise ValueError(f"Phase '{self.name}': unknown mode '{self.mode}'")
        if self.mode == "mixed":
            if not self.domains:
                raise ValueError(f"Phase '{self.name}': mixed mode requires 'domains'")
        elif not self.domain:
            raise ValueError(f"Phase '{self.name}': mode '{self.mode}' requires 'domain'")
        if self.steps <= 0:
            raise ValueError(f"Phase '{self.name}': steps must be positive")


class _NovelCursor:
    """Hands out never-before-used rows from a pre-shuffled pool.

    Replaces a per-batch ``[i for i in pool if i not in used]`` rescan. That
    scan is O(pool) per novel batch, and at headline scale (160k rows,
    ~1160 novel batches across the stream) it costs ~10^8 membership tests
    of pure startup latency before step 0. Shuffling once and walking a
    cursor gives the same guarantee — each row used at most once, globally,
    across every phase sharing the ``used`` set — in O(pool) total.
    """

    def __init__(
        self, rng: np.random.Generator, pool: list[int], used: set[int]
    ) -> None:
        self._order = [int(i) for i in rng.permutation(pool)]
        self._pos = 0
        self._used = used

    def take(self, n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n and self._pos < len(self._order):
            idx = self._order[self._pos]
            self._pos += 1
            if idx in self._used:
                continue
            self._used.add(idx)
            out.append(idx)
        if len(out) < n:
            # Return what we consumed; the schedule is unusable either way.
            raise ValueError(
                "Not enough unused rows for a novel batch — increase the domain "
                "row cap (data.max_rows_per_domain) or reduce novel steps."
            )
        return out


@dataclass
class StreamConfig:
    enabled: bool = False
    seed: Optional[int] = None  # falls back to train.seed
    phases: list[PhaseSpec] = field(default_factory=list)


def parse_stream_config(raw: Optional[dict[str, Any]]) -> StreamConfig:
    """Build a StreamConfig from a plain (Hydra-resolved) dict."""
    if not raw:
        return StreamConfig()
    phases = []
    for p in raw.get("phases", []):
        p = dict(p)
        rec = p.pop("recurrence", None)
        phases.append(
            PhaseSpec(**p, recurrence=RecurrenceSpec(**rec) if rec else None)
        )
    return StreamConfig(
        enabled=bool(raw.get("enabled", False)),
        seed=raw.get("seed"),
        phases=phases,
    )


class StreamSchedule:
    """Deterministic step → batch-indices mapping over a phased stream."""

    def __init__(
        self,
        cfg: StreamConfig,
        dataset: Any,  # HivemindHFDataset (needs indices_for_domain / indices_by_bucket)
        batch_size: int,
        seed: int,
        data_identity: Optional[dict[str, Any]] = None,
    ) -> None:
        if not cfg.phases:
            raise ValueError("StreamSchedule requires at least one phase")
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.seed = int(cfg.seed if cfg.seed is not None else seed)
        # Folded into config_digest so the resume guard notices a changed
        # corpus, not just a changed phase list. See config_digest().
        self.data_identity = data_identity or {}

        self.total_steps = sum(p.steps for p in cfg.phases)
        self._phase_starts: list[int] = []
        acc = 0
        for p in cfg.phases:
            self._phase_starts.append(acc)
            acc += p.steps

        # Per-phase chosen buckets: {phase_name: {bucket_key: [indices]}}.
        self._phase_buckets: dict[str, dict[str, list[int]]] = {}
        self._batches: list[list[int]] = []
        self._bucket_overrides: dict[int, str] = {}  # step -> synthetic bucket id
        self._build(dataset)

    # -- construction ------------------------------------------------------

    def _rng(self, phase_index: int, tag: int = 0) -> np.random.Generator:
        return np.random.default_rng(
            (self.seed * 1_000_003 + phase_index * 7919 + tag) % (2**63)
        )

    def _build(self, dataset: Any) -> None:
        used: set[int] = set()  # rows claimed by recurrent subsets / novel draws

        # Pass 1: recurrent phases claim bucket subsets (so background/novel
        # draws exclude them); reuse_from resolves against earlier phases.
        for pi, phase in enumerate(self.cfg.phases):
            if phase.mode != "recurrent":
                continue
            rec = phase.recurrence or RecurrenceSpec()
            if rec.reuse_from:
                if rec.reuse_from not in self._phase_buckets:
                    raise ValueError(
                        f"Phase '{phase.name}': reuse_from='{rec.reuse_from}' must "
                        "name an EARLIER recurrent phase"
                    )
                self._phase_buckets[phase.name] = self._phase_buckets[rec.reuse_from]
                continue
            all_buckets = dataset.indices_by_bucket(phase.domain)
            eligible = sorted(
                k for k, idxs in all_buckets.items() if len(idxs) >= rec.samples_per_bucket
            )
            if len(eligible) < rec.num_buckets:
                raise ValueError(
                    f"Phase '{phase.name}': needs {rec.num_buckets} buckets with "
                    f"≥{rec.samples_per_bucket} rows in domain '{phase.domain}', "
                    f"found {len(eligible)}"
                )
            rng = self._rng(pi, tag=1)
            chosen = list(rng.choice(eligible, size=rec.num_buckets, replace=False))
            subsets: dict[str, list[int]] = {}
            for key in chosen:
                rows = list(all_buckets[key])
                rng.shuffle(rows)
                subset = rows[: rec.samples_per_bucket]
                subsets[str(key)] = subset
                used.update(subset)
            self._phase_buckets[phase.name] = subsets

        # Pass 2: generate the batch sequence phase by phase.
        step = 0
        for pi, phase in enumerate(self.cfg.phases):
            builder = {
                "random": self._build_random,
                "recurrent": self._build_recurrent,
                "novel": self._build_novel,
                "mixed": self._build_mixed,
            }[phase.mode]
            batches = builder(pi, phase, dataset, used, step)
            assert len(batches) == phase.steps
            self._batches.extend(batches)
            step += phase.steps

    def _background_pool(
        self,
        dataset: Any,
        domain: str,
        used: set[int],
        exclude_rows: Optional[set[int]] = None,
    ) -> list[int]:
        blocked = used if not exclude_rows else (used | exclude_rows)
        pool = [i for i in dataset.indices_for_domain(domain) if i not in blocked]
        if not pool:  # tiny datasets: fall back to the full domain
            pool = list(dataset.indices_for_domain(domain))
        return pool

    def _iid_batches(
        self, rng: np.random.Generator, pool: list[int], n_batches: int
    ) -> list[list[int]]:
        """Without-replacement draws, reshuffling when the pool is exhausted."""
        out: list[list[int]] = []
        perm: list[int] = []
        for _ in range(n_batches):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                if not perm:
                    perm = list(rng.permutation(pool))
                batch.append(perm.pop())
            out.append(batch)
        return out

    def _build_random(
        self, pi: int, phase: PhaseSpec, dataset: Any, used: set[int], start: int
    ) -> list[list[int]]:
        rng = self._rng(pi, tag=2)
        pool = self._background_pool(dataset, phase.domain, used)
        return self._iid_batches(rng, pool, phase.steps)

    def _build_recurrent(
        self, pi: int, phase: PhaseSpec, dataset: Any, used: set[int], start: int
    ) -> list[list[int]]:
        rec = phase.recurrence or RecurrenceSpec()
        subsets = self._phase_buckets[phase.name]
        bucket_keys = sorted(subsets)
        period = rec.revisit_period or len(bucket_keys)
        if period < len(bucket_keys):
            raise ValueError(
                f"Phase '{phase.name}': revisit_period ({period}) < num_buckets "
                f"({len(bucket_keys)})"
            )
        # Spread bucket slots evenly across the period.
        slot_of = {
            (b * period) // len(bucket_keys): bucket_keys[b]
            for b in range(len(bucket_keys))
        }

        rng = self._rng(pi, tag=3)
        exclude: Optional[set[int]] = None
        if rec.exclude_chosen_buckets_from_background:
            all_buckets = dataset.indices_by_bucket(phase.domain)
            # Every row of a chosen bucket, not just the sampled subset —
            # the controller keys repetition on the bucket, not the row.
            exclude = {i for k in subsets for i in all_buckets.get(k, ())}
        pool = self._background_pool(dataset, phase.domain, used, exclude)
        novel = _NovelCursor(rng, pool, used)
        visits: dict[str, int] = {k: 0 for k in bucket_keys}
        batches: list[list[int]] = []
        novel_serial = 0
        for t in range(phase.steps):
            slot = t % period
            key = slot_of.get(slot)
            if key is not None:
                subset = subsets[key]
                v = visits[key]
                visits[key] = v + 1
                batch = [
                    subset[(v * self.batch_size + j) % len(subset)]
                    for j in range(self.batch_size)
                ]
            elif rec.novel_fraction > 0 and rng.random() < rec.novel_fraction:
                batch = novel.take(self.batch_size)
                self._bucket_overrides[start + t] = (
                    f"novel_{phase.name}_{novel_serial}"
                )
                novel_serial += 1
            else:
                batch = self._iid_batches(rng, pool, 1)[0]
            batches.append(batch)
        return batches

    def _build_novel(
        self, pi: int, phase: PhaseSpec, dataset: Any, used: set[int], start: int
    ) -> list[list[int]]:
        rng = self._rng(pi, tag=4)
        pool = list(dataset.indices_for_domain(phase.domain))
        novel = _NovelCursor(rng, pool, used)
        batches = []
        for t in range(phase.steps):
            batch = novel.take(self.batch_size)
            self._bucket_overrides[start + t] = f"novel_{phase.name}_{t}"
            batches.append(batch)
        return batches

    def _build_mixed(
        self, pi: int, phase: PhaseSpec, dataset: Any, used: set[int], start: int
    ) -> list[list[int]]:
        rng = self._rng(pi, tag=5)
        domains = list(phase.domains)
        weights = phase.weights or [1.0 / len(domains)] * len(domains)
        if len(weights) != len(domains):
            raise ValueError(f"Phase '{phase.name}': weights/domains length mismatch")
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        pools = {d: self._background_pool(dataset, d, used) for d in domains}
        perms: dict[str, list[int]] = {d: [] for d in domains}
        batches = []
        for _ in range(phase.steps):
            d = domains[int(rng.choice(len(domains), p=w))]
            batch: list[int] = []
            while len(batch) < self.batch_size:
                if not perms[d]:
                    perms[d] = list(rng.permutation(pools[d]))
                batch.append(perms[d].pop())
            batches.append(batch)
        return batches

    # -- query API ---------------------------------------------------------

    def indices_for_step(self, step: int) -> list[int]:
        return [int(i) for i in self._batches[step]]

    def phase_index_at(self, step: int) -> int:
        from bisect import bisect_right

        return bisect_right(self._phase_starts, step) - 1

    def phase_at(self, step: int) -> PhaseSpec:
        return self.cfg.phases[self.phase_index_at(step)]

    @property
    def phase_boundaries(self) -> list[tuple[int, str]]:
        return [
            (start, phase.name)
            for start, phase in zip(self._phase_starts, self.cfg.phases)
        ]

    def bucket_override(self, step: int) -> Optional[str]:
        """Synthetic unique bucket id for novel batches, else None."""
        return self._bucket_overrides.get(step)

    def phase_buckets(self, phase_name: str) -> dict[str, list[int]]:
        return {k: list(v) for k, v in self._phase_buckets.get(phase_name, {}).items()}

    def config_digest(self) -> str:
        """Fingerprint of everything that determines the batch sequence.

        Stored in checkpoint metadata and compared on resume. It must cover
        the *data* as well as the phase list: which repos are loaded, how
        each domain's buckets are keyed, and how the row cap is applied all
        change which rows land in which batch, and previously a run could
        swap its corpus and still pass the resume guard.
        """
        payload = {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "phases": [asdict(p) for p in self.cfg.phases],
            "data": self.data_identity,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]


class StreamBatchSampler(Sampler[list[int]]):
    """Batch sampler that replays a StreamSchedule from ``start_step``."""

    def __init__(self, schedule: StreamSchedule, start_step: int = 0) -> None:
        self.schedule = schedule
        self.start_step = int(start_step)

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.schedule.total_steps):
            yield self.schedule.indices_for_step(step)

    def __len__(self) -> int:
        return self.schedule.total_steps - self.start_step

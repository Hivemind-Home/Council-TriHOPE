"""Every shipped config must build a StreamSchedule against the real data shape.

This is the regression test for the class of bug that shipped in
``stream_small``/``stream_headline``: ``code_recurrent`` asked for 8 buckets
while the published ``code-layerC-200k`` corpus has exactly 3 distinct
``bucket_id`` values, so ``StreamSchedule`` raised at construction — before
step 0, on every run in Groups A/B/C/D. It went unnoticed because
``stream_smoke`` streams math only (15 buckets), so the smoke test passed.

The fix here is to check *config against data shape* without downloading
anything: a fake dataset reproduces the measured per-domain bucket
cardinalities, and each config's stream is built against it. Milliseconds,
no network, and it fails loudly the moment a config asks for more buckets
than the corpus can supply.

Cardinalities below were measured directly from the published parquets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hivemind.data.hf_loader import normalize_domain_specs
from hivemind.data.stream import StreamSchedule, parse_stream_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

#: Distinct ``bucket_id`` values per domain in the published Layer-C repos.
#: ``bucket_id`` is ``{domain}_{subdomain}_{difficulty}``; code has a single
#: subdomain, which is why its native count is so low.
NATIVE_BUCKETS = {"general": 27, "code": 3, "math": 15, "medical": 17}

#: Bucket count when a domain declares ``bucket_columns``. Keyed by the
#: tuple of columns so a config that picks a different combination is not
#: silently validated against the wrong number.
COMPOSITE_BUCKETS = {
    ("code", ("subdomain", "difficulty", "source")): 12,
    ("code", ("subdomain", "difficulty", "source", "task_type")): 19,
    ("math", ("subdomain", "difficulty", "source")): 60,
    ("medical", ("subdomain", "difficulty", "source")): 60,
    ("general", ("subdomain", "difficulty", "source")): 27,  # general has 1 source
}

#: Generous enough that novel phases never exhaust their pool; these are
#: index lists only, so the cost is trivial.
ROWS_PER_BUCKET = 500


def bucket_count(spec) -> int:
    if not spec.bucket_columns:
        if spec.name not in NATIVE_BUCKETS:
            pytest.fail(
                f"Domain '{spec.name}' has no measured bucket count. Add it to "
                "NATIVE_BUCKETS (query the repo's bucket_id cardinality first)."
            )
        return NATIVE_BUCKETS[spec.name]
    key = (spec.name, tuple(spec.bucket_columns))
    if key not in COMPOSITE_BUCKETS:
        pytest.fail(
            f"Domain '{spec.name}' declares bucket_columns={spec.bucket_columns}, "
            "whose cardinality has not been measured against the real corpus. "
            "Measure it and add it to COMPOSITE_BUCKETS."
        )
    return COMPOSITE_BUCKETS[key]


class FakeShapedDataset:
    """Reproduces the real per-domain bucket shape, with no data behind it."""

    def __init__(self, specs, rows_per_bucket: int = ROWS_PER_BUCKET) -> None:
        self._domains: dict[str, range] = {}
        self._buckets: dict[str, dict[str, list[int]]] = {}
        cursor = 0
        for spec in specs:
            n = bucket_count(spec)
            total = n * rows_per_bucket
            self._domains[spec.name] = range(cursor, cursor + total)
            self._buckets[spec.name] = {
                f"{spec.name}_b{b}": [
                    cursor + b * rows_per_bucket + r for r in range(rows_per_bucket)
                ]
                for b in range(n)
            }
            cursor += total

    def indices_for_domain(self, name: str) -> range:
        if name not in self._domains:
            raise KeyError(f"Domain '{name}' not loaded (have {sorted(self._domains)})")
        return self._domains[name]

    def indices_by_bucket(self, name: str) -> dict[str, list[int]]:
        if name not in self._buckets:
            raise KeyError(f"Domain '{name}' not loaded (have {sorted(self._buckets)})")
        return self._buckets[name]


def stream_config_paths() -> list[Path]:
    out = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        stream = cfg.get("stream")
        if stream and stream.get("enabled"):
            out.append(path)
    return out


STREAM_CONFIGS = stream_config_paths()


def test_some_stream_configs_were_discovered():
    """Guard against the glob silently matching nothing."""
    assert STREAM_CONFIGS, f"no stream-enabled configs found under {CONFIG_DIR}"


@pytest.fixture(params=STREAM_CONFIGS, ids=lambda p: p.stem)
def stream_config(request):
    path = request.param
    return path, OmegaConf.load(path)


def _specs(cfg):
    raw = OmegaConf.to_container(cfg.data.domains, resolve=True)
    return normalize_domain_specs(raw, cfg.data.get("repo_prefix", "hivemind-research"))


def test_stream_builds_against_real_bucket_shape(stream_config):
    """The check that would have caught the code-bucket bug."""
    path, cfg = stream_config
    specs = _specs(cfg)
    dataset = FakeShapedDataset(specs)
    stream = parse_stream_config(OmegaConf.to_container(cfg.stream, resolve=True))

    schedule = StreamSchedule(
        stream,
        dataset,
        batch_size=int(cfg.data.batch_size),
        seed=int(cfg.data.get("seed", 42)),
    )
    assert schedule.total_steps == sum(p.steps for p in stream.phases)
    # Every step must produce a full batch of in-range indices.
    for step in (0, schedule.total_steps // 2, schedule.total_steps - 1):
        batch = schedule.indices_for_step(step)
        assert len(batch) == int(cfg.data.batch_size), f"{path.stem} step {step}"


def test_every_phase_domain_is_loaded(stream_config):
    """A phase naming an undeclared domain fails at step 0 today."""
    path, cfg = stream_config
    declared = {s.name for s in _specs(cfg)}
    for phase in cfg.stream.phases:
        used = list(phase.get("domains") or []) or [phase.get("domain")]
        for name in used:
            assert name in declared, (
                f"{path.stem}: phase '{phase['name']}' uses domain '{name}', "
                f"which data.domains does not declare ({sorted(declared)})"
            )


def test_mixed_phase_weights_match_domains(stream_config):
    path, cfg = stream_config
    for phase in cfg.stream.phases:
        if phase.get("mode") != "mixed":
            continue
        weights = phase.get("weights")
        if weights is None:
            continue
        assert len(weights) == len(phase["domains"]), (
            f"{path.stem}: phase '{phase['name']}' has {len(weights)} weights for "
            f"{len(phase['domains'])} domains"
        )


def test_declared_train_steps_match_the_stream(stream_config):
    """``train.steps: null`` derives from the stream; an explicit value must agree."""
    path, cfg = stream_config
    declared = cfg.get("train", {}).get("steps")
    if declared is None:
        return
    total = sum(int(p["steps"]) for p in cfg.stream.phases)
    assert int(declared) == total, (
        f"{path.stem}: train.steps={declared} but the stream totals {total}"
    )


def test_teacher_ids_cover_every_domain(stream_config):
    """A domain absent from teacher_ids falls back to teacher 0 silently."""
    path, cfg = stream_config
    teacher_ids = cfg.get("teachers", {}).get("teacher_ids")
    if teacher_ids is None:
        return
    for spec in _specs(cfg):
        assert spec.name in teacher_ids, (
            f"{path.stem}: domain '{spec.name}' has no teachers.teacher_ids entry, "
            "so metadata routing cannot resolve it by name"
        )


class TestTheFixtureItself:
    """The fake must actually reproduce the shape it claims to."""

    def test_code_native_is_three_buckets(self):
        specs = normalize_domain_specs([{"name": "code", "layer_c": "org/code"}])
        ds = FakeShapedDataset(specs)
        assert len(ds.indices_by_bucket("code")) == 3

    def test_code_composite_is_twelve_buckets(self):
        specs = normalize_domain_specs([
            {
                "name": "code",
                "layer_c": "org/code",
                "bucket_columns": ["subdomain", "difficulty", "source"],
            }
        ])
        ds = FakeShapedDataset(specs)
        assert len(ds.indices_by_bucket("code")) == 12

    def test_native_code_cannot_supply_eight_buckets(self):
        """The original bug, pinned: 8 requested, 3 available."""
        specs = normalize_domain_specs([{"name": "code", "layer_c": "org/code"}])
        stream = parse_stream_config({
            "enabled": True,
            "phases": [{
                "name": "code_recurrent",
                "domain": "code",
                "steps": 10,
                "mode": "recurrent",
                "recurrence": {"num_buckets": 8, "samples_per_bucket": 16},
            }],
        })
        with pytest.raises(ValueError, match="needs 8 buckets"):
            StreamSchedule(stream, FakeShapedDataset(specs), batch_size=2, seed=1)

    def test_composite_code_can_supply_eight_buckets(self):
        specs = normalize_domain_specs([{
            "name": "code",
            "layer_c": "org/code",
            "bucket_columns": ["subdomain", "difficulty", "source"],
        }])
        stream = parse_stream_config({
            "enabled": True,
            "phases": [{
                "name": "code_recurrent",
                "domain": "code",
                "steps": 10,
                "mode": "recurrent",
                "recurrence": {"num_buckets": 8, "samples_per_bucket": 16},
            }],
        })
        schedule = StreamSchedule(stream, FakeShapedDataset(specs), batch_size=2, seed=1)
        assert schedule.total_steps == 10

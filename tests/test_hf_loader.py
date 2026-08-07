"""Tests for HivemindHFDataset.

The HF ``load_dataset`` call is monkeypatched so the test doesn't hit the
network. We hand back in-memory lists that behave like ``datasets.Dataset``
for the fields the loader reads (iteration + ``column_names``).
"""

from __future__ import annotations

import pytest

from hivemind.data.hf_loader import (
    DatasetLoadError,
    HivemindHFConfig,
    HivemindHFDataset,
    normalize_domain_specs,
)


class _FakeHFDataset(list):
    """Duck-typed stand-in for datasets.Dataset in tests."""

    def __init__(self, rows):
        super().__init__(rows)
        if rows:
            self.column_names = list(rows[0].keys())
        else:
            self.column_names = []


@pytest.fixture
def fake_loader(monkeypatch, hf_sample_rows):
    """Install a fake ``load_dataset`` keyed by repo name."""
    repos: dict[str, _FakeHFDataset] = {
        "hivemind-research/code-layerA-final": _FakeHFDataset([
            {k: r[k] for k in ("sample_id", "input_text", "target_text", "bucket_id", "domain")}
            for r in hf_sample_rows if r["domain"] == "code"
        ]),
        "hivemind-research/code-layerB-final": _FakeHFDataset([
            {
                k: r[k]
                for k in ("sample_id", "teacher_id", "teacher_output_text", "teacher_logits_path")
            }
            for r in hf_sample_rows if r["domain"] == "code"
        ]),
    }

    def fake_load_dataset(repo, split, cache_dir=None, streaming=False):
        if repo not in repos:
            raise FileNotFoundError(repo)
        return repos[repo]

    # The loader does ``from datasets import load_dataset`` inside the
    # method body, so the only patch that matters is the one on the
    # ``datasets`` module itself.
    import sys
    import types

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return repos


def test_loads_and_joins_layers(fake_loader):
    cfg = HivemindHFConfig(
        domains=["code"],
        tokenizer_name="whitespace",
        max_seq_len=32,
    )
    ds = HivemindHFDataset(cfg)
    assert len(ds) == 5
    row = ds[0]
    assert "input_text" in row and "teacher_output_text" in row
    assert row["domain"] == "code"


def test_missing_repo_raises_under_strict_load(fake_loader):
    """The default must name the repo that failed, not silently drop it."""
    cfg = HivemindHFConfig(
        domains=["math"],  # math-layerB isn't in the fake repos
        tokenizer_name="whitespace",
        max_seq_len=32,
    )
    with pytest.raises(DatasetLoadError, match="math-layerB-final"):
        HivemindHFDataset(cfg)


def test_missing_layer_a_skips_domain(fake_loader, caplog):
    cfg = HivemindHFConfig(
        domains=["math"],  # math-layerA isn't in the fake repos → skipped
        tokenizer_name="whitespace",
        max_seq_len=32,
        strict_load=False,
    )
    with pytest.raises(RuntimeError, match="no usable rows"):
        HivemindHFDataset(cfg)


def test_max_rows_per_domain_truncates(fake_loader):
    cfg = HivemindHFConfig(
        domains=["code"],
        tokenizer_name="whitespace",
        max_seq_len=32,
        max_rows_per_domain=2,
    )
    ds = HivemindHFDataset(cfg)
    assert len(ds) == 2


# -- Layer C path -----------------------------------------------------------


class _FakeArrowDataset(_FakeHFDataset):
    """Adds select() and column access, mimicking an Arrow-backed Dataset."""

    def select(self, indices):
        return _FakeArrowDataset([list.__getitem__(self, i) for i in indices])

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in list(self)]
        return list.__getitem__(self, key)


def _layer_c_rows(domain: str, n: int):
    # ``subdomain`` is deliberately constant, mirroring the published code
    # corpus where every row shares one subdomain and ``bucket_id`` therefore
    # collapses to difficulty alone. That is what ``bucket_columns`` exists
    # to work around.
    return [
        {
            "sample_id": f"{domain}-{i}",
            "input_text": f"{domain} question {i}",
            "target_text": f"answer {i}",
            "has_gold_label": i % 2 == 0,
            "domain": domain,
            "subdomain": "core",
            "difficulty": ["easy", "medium", "hard"][i % 3],
            "source": f"src{i % 2}",
            "bucket_id": f"{domain}_bucket_{i % 3}",
            "teacher_id": f"{domain}_teacher",
            "teacher_output_text": f"teacher says {i}",
            "teacher_logits_path": "",
            "teacher_confidence": 0.9,
        }
        for i in range(n)
    ]


@pytest.fixture
def fake_layer_c(monkeypatch):
    repos = {
        "hivemind-research/code-layerC-200k": {
            "train": _FakeArrowDataset(_layer_c_rows("code", 9)),
            "validation": _FakeArrowDataset(_layer_c_rows("code", 4)),
        },
        "hivemind-research/math-layerC-200K": {
            "train": _FakeArrowDataset(_layer_c_rows("math", 6)),
        },
    }

    def fake_load_dataset(repo, split, cache_dir=None, streaming=False):
        if repo not in repos or split not in repos[repo]:
            raise FileNotFoundError(f"{repo}:{split}")
        return repos[repo][split]

    import sys
    import types

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return repos


def _spec_cfg(**kwargs) -> HivemindHFConfig:
    defaults = dict(
        domains=[
            {"name": "code", "layer_c": "hivemind-research/code-layerC-200k"},
            {"name": "math", "layer_c": "hivemind-research/math-layerC-200K"},
        ],
        tokenizer_name="whitespace",
        max_seq_len=32,
    )
    defaults.update(kwargs)
    return HivemindHFConfig(**defaults)


def test_layer_c_direct_load(fake_layer_c):
    ds = HivemindHFDataset(_spec_cfg())
    assert len(ds) == 15
    assert ds.domains == ["code", "math"]
    assert ds[0]["domain"] == "code"
    assert ds[9]["domain"] == "math"
    assert ds[9]["teacher_output_text"].startswith("teacher says")


def test_layer_c_split_selection(fake_layer_c):
    ds = HivemindHFDataset(
        _spec_cfg(
            domains=[{"name": "code", "layer_c": "hivemind-research/code-layerC-200k"}],
            split="validation",
        )
    )
    assert len(ds) == 4


def test_layer_c_per_domain_cap(fake_layer_c):
    ds = HivemindHFDataset(
        _spec_cfg(
            domains=[
                {"name": "code", "layer_c": "hivemind-research/code-layerC-200k", "max_rows": 3},
                {"name": "math", "layer_c": "hivemind-research/math-layerC-200K"},
            ],
        )
    )
    assert len(ds) == 9  # 3 code + 6 math
    assert len(list(ds.indices_for_domain("code"))) == 3


def test_stream_accessors(fake_layer_c):
    ds = HivemindHFDataset(_spec_cfg())
    code_range = ds.indices_for_domain("code")
    assert code_range == range(0, 9)
    assert ds.indices_for_domain("math") == range(9, 15)

    buckets = ds.indices_by_bucket("code")
    assert set(buckets) == {"code_bucket_0", "code_bucket_1", "code_bucket_2"}
    assert sorted(i for lst in buckets.values() for i in lst) == list(range(9))

    gold = ds.gold_indices("math")
    assert gold == [9, 11, 13]  # even local indices offset by 9


def test_unknown_spec_key_raises(fake_layer_c):
    with pytest.raises(ValueError, match="Unknown DomainSpec keys"):
        HivemindHFDataset(
            _spec_cfg(domains=[{"name": "code", "layerc": "typo-key"}])
        )


def test_string_and_dict_specs_mix(fake_loader, fake_layer_c, monkeypatch):
    # fake_layer_c overwrites the datasets module last; merge both fakes.
    import sys

    layer_ab = dict(fake_loader)
    layer_c = fake_layer_c

    def merged_load(repo, split, cache_dir=None, streaming=False):
        if repo in layer_c and split in layer_c[repo]:
            return layer_c[repo][split]
        if repo in layer_ab:
            return layer_ab[repo]
        raise FileNotFoundError(repo)

    sys.modules["datasets"].load_dataset = merged_load

    ds = HivemindHFDataset(
        _spec_cfg(
            domains=[
                "code",  # legacy A+B join
                {"name": "math", "layer_c": "hivemind-research/math-layerC-200K"},
            ]
        )
    )
    assert ds.domains == ["code", "math"]
    assert len(ds) == 5 + 6


# -- bucket_columns: composite bucket keys -----------------------------------


class TestBucketColumns:
    """A degenerate published ``bucket_id`` must be overridable from config.

    The code Layer-C corpus assigns one subdomain to every row, so its
    ``bucket_id`` yields only 3 distinct values — fewer than a recurrent
    stream phase requires. ``bucket_columns`` synthesizes a finer key.
    """

    def test_default_uses_published_bucket_id(self, fake_layer_c):
        ds = HivemindHFDataset(_spec_cfg())
        assert set(ds.indices_by_bucket("code")) == {
            "code_bucket_0", "code_bucket_1", "code_bucket_2"
        }

    def test_composite_key_increases_cardinality(self, fake_layer_c):
        ds = HivemindHFDataset(
            _spec_cfg(
                domains=[{
                    "name": "code",
                    "layer_c": "hivemind-research/code-layerC-200k",
                    "bucket_columns": ["subdomain", "difficulty", "source"],
                }]
            )
        )
        buckets = ds.indices_by_bucket("code")
        assert set(buckets) == {
            "core_easy_src0", "core_easy_src1",
            "core_medium_src0", "core_medium_src1",
            "core_hard_src0", "core_hard_src1",
        }
        # Every row is still assigned exactly once.
        assert sorted(i for lst in buckets.values() for i in lst) == list(range(9))

    def test_missing_column_raises_instead_of_collapsing(self, fake_layer_c):
        """Otherwise every row keys to 'None' and one bucket looks legitimate."""
        ds = HivemindHFDataset(
            _spec_cfg(
                domains=[{
                    "name": "code",
                    "layer_c": "hivemind-research/code-layerC-200k",
                    "bucket_columns": ["subdomain", "nonexistent_column"],
                }]
            )
        )
        with pytest.raises(ValueError, match="nonexistent_column"):
            ds.indices_by_bucket("code")

    def test_normalize_accepts_and_stringifies_bucket_columns(self):
        specs = normalize_domain_specs(
            [{"name": "code", "layer_c": "org/repo", "bucket_columns": ["a", "b"]}]
        )
        assert specs[0].bucket_columns == ["a", "b"]

    def test_other_domains_keep_their_published_buckets(self, fake_layer_c):
        ds = HivemindHFDataset(
            _spec_cfg(
                domains=[
                    {
                        "name": "code",
                        "layer_c": "hivemind-research/code-layerC-200k",
                        "bucket_columns": ["subdomain", "difficulty", "source"],
                    },
                    {"name": "math", "layer_c": "hivemind-research/math-layerC-200K"},
                ]
            )
        )
        assert len(ds.indices_by_bucket("code")) == 6
        assert set(ds.indices_by_bucket("math")) == {
            "math_bucket_0", "math_bucket_1", "math_bucket_2"
        }


# -- shuffle_before_cap ------------------------------------------------------


class _ShufflableArrow(_FakeArrowDataset):
    def shuffle(self, seed=0):
        rows = list(self)
        # Deterministic, seed-dependent, and definitely not the identity.
        rows = rows[seed % max(1, len(rows)):] + rows[: seed % max(1, len(rows))]
        return _ShufflableArrow(rows)

    def select(self, indices):
        return _ShufflableArrow([list.__getitem__(self, i) for i in indices])


class TestShuffleBeforeCap:
    """Capping the head of an unshuffled parquet starves bucket cardinality."""

    @pytest.fixture
    def shufflable(self, monkeypatch):
        import sys
        import types

        repos = {"org/head-sorted": {"train": _ShufflableArrow(_layer_c_rows("code", 9))}}

        def fake_load_dataset(repo, split, cache_dir=None, streaming=False):
            if repo not in repos or split not in repos[repo]:
                raise FileNotFoundError(f"{repo}:{split}")
            return repos[repo][split]

        mod = types.ModuleType("datasets")
        mod.load_dataset = fake_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", mod)
        return repos

    def _ids(self, **kwargs):
        ds = HivemindHFDataset(
            _spec_cfg(
                domains=[{"name": "code", "layer_c": "org/head-sorted"}],
                max_rows_per_domain=3,
                **kwargs,
            )
        )
        return [ds[i]["sample_id"] for i in range(len(ds))]

    def test_disabled_takes_the_head(self, shufflable):
        assert self._ids(shuffle_before_cap=False) == ["code-0", "code-1", "code-2"]

    def test_enabled_samples_elsewhere(self, shufflable):
        ids = self._ids(shuffle_before_cap=True, shuffle_seed=4)
        assert len(ids) == 3
        assert ids != ["code-0", "code-1", "code-2"]

    def test_enabled_is_deterministic_in_the_seed(self, shufflable):
        assert self._ids(shuffle_before_cap=True, shuffle_seed=4) == self._ids(
            shuffle_before_cap=True, shuffle_seed=4
        )


def test_streaming_dataset_without_columns_is_rejected(monkeypatch):
    """The loader indexes rows positionally, so an IterableDataset cannot work."""
    import sys
    import types

    class _NoColumns(list):
        column_names = None

    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda repo, split, cache_dir=None, streaming=False: _NoColumns(
        _layer_c_rows("code", 3)
    )
    monkeypatch.setitem(sys.modules, "datasets", mod)

    with pytest.raises(ValueError, match="streaming"):
        HivemindHFDataset(
            _spec_cfg(domains=[{"name": "code", "layer_c": "org/streamed"}])
        )

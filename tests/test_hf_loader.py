"""Tests for HivemindHFDataset.

The HF ``load_dataset`` call is monkeypatched so the test doesn't hit the
network. We hand back in-memory lists that behave like ``datasets.Dataset``
for the fields the loader reads (iteration + ``column_names``).
"""

from __future__ import annotations

import pytest

from hivemind.data.hf_loader import HivemindHFConfig, HivemindHFDataset


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
            {k: r[k] for k in ("sample_id", "teacher_id", "teacher_output_text", "teacher_logits_path")}
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


def test_missing_layer_a_skips_domain(fake_loader, caplog):
    cfg = HivemindHFConfig(
        domains=["math"],  # math-layerA isn't in the fake repos → skipped
        tokenizer_name="whitespace",
        max_seq_len=32,
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

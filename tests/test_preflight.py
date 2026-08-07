"""Preflight: report every config/data mismatch at once, before GPU time.

Each case here corresponds to a failure that actually shipped: a bucket
request the corpus could not satisfy (late crash), teacher ids that matched
nothing (no error at all), an exact-match probe set of 3 (no error at all),
and a domain that silently failed to load.
"""

from __future__ import annotations

import pytest

from hivemind.data.preflight import PreflightError, validate, validate_or_raise


class FakeDataset:
    def __init__(self, buckets: dict[str, int], gold: dict[str, int] | None = None,
                 teacher_ids: dict[str, str] | None = None):
        """``buckets`` maps domain -> (n_buckets, rows_per_bucket) via a tuple."""
        self._buckets: dict[str, dict[str, list[int]]] = {}
        self._rows: dict[int, dict] = {}
        self._ranges: dict[str, range] = {}
        self._gold = gold or {}
        cursor = 0
        for name, (n, per) in buckets.items():
            start = cursor
            self._buckets[name] = {}
            for b in range(n):
                idxs = list(range(cursor, cursor + per))
                self._buckets[name][f"{name}_b{b}"] = idxs
                for i in idxs:
                    self._rows[i] = {
                        "teacher_id": (teacher_ids or {}).get(name, f"{name}_teacher")
                    }
                cursor += per
            self._ranges[name] = range(start, cursor)

    @property
    def domains(self):
        return list(self._ranges)

    def indices_for_domain(self, name):
        return self._ranges[name]

    def indices_by_bucket(self, name):
        return self._buckets[name]

    def gold_indices(self, name):
        return list(range(self._gold.get(name, 0)))

    def __getitem__(self, i):
        return self._rows[i]


def _data(domains, **kw):
    return {"domains": domains, "batch_size": 2, **kw}


def _recurrent(name, domain, num_buckets=8, samples=16, steps=100, **rec):
    return {
        "name": name, "domain": domain, "steps": steps, "mode": "recurrent",
        "recurrence": {
            "num_buckets": num_buckets, "samples_per_bucket": samples, **rec
        },
    }


class TestBucketCardinality:
    """The bug that shipped: 8 buckets requested, 3 available."""

    def test_insufficient_buckets_is_an_error(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [_recurrent("code_recurrent", "code")]},
            dataset=FakeDataset({"code": (3, 500)}),
        )
        assert not rep.ok
        joined = " ".join(rep.errors)
        assert "code_recurrent" in joined and "needs 8 buckets" in joined

    def test_the_error_suggests_both_remedies(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [_recurrent("code_recurrent", "code")]},
            dataset=FakeDataset({"code": (3, 500)}),
        )
        joined = " ".join(rep.errors)
        assert "lower num_buckets to 3" in joined
        assert "bucket_columns" in joined

    def test_sufficient_buckets_pass(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [_recurrent("code_recurrent", "code")]},
            dataset=FakeDataset({"code": (12, 500)}),
        )
        assert rep.ok, rep.render()

    def test_buckets_below_the_row_threshold_do_not_count(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [_recurrent("code_recurrent", "code", samples=16)]},
            dataset=FakeDataset({"code": (12, 4)}),  # 12 buckets, only 4 rows each
        )
        assert not rep.ok
        assert "0 of 12 qualify" in " ".join(rep.errors)

    def test_reuse_from_phases_are_not_rechecked(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [
                _recurrent("code_recurrent", "code", num_buckets=8),
                _recurrent("code_revisit", "code", num_buckets=8,
                           reuse_from="code_recurrent"),
            ]},
            dataset=FakeDataset({"code": (12, 500)}),
        )
        assert rep.ok, rep.render()


class TestAllProblemsAtOnce:
    """First-failure reporting hides the rest behind it."""

    def test_multiple_errors_are_all_reported(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [
                _recurrent("code_recurrent", "code"),
                {"name": "ghost", "domain": "medical", "steps": 10, "mode": "random"},
            ]},
            dataset=FakeDataset({"code": (3, 500)}),
        )
        assert len(rep.errors) >= 2
        joined = " ".join(rep.errors)
        assert "needs 8 buckets" in joined
        assert "does not declare" in joined

    def test_render_numbers_the_errors(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [_recurrent("code_recurrent", "code")]},
            dataset=FakeDataset({"code": (3, 500)}),
        )
        assert "1 blocking problem" in rep.render()


class TestTeacherIds:
    def test_a_mismatched_teacher_id_is_an_error_naming_the_real_value(self):
        rep = validate(
            data_cfg=_data([{"name": "math", "layer_c": "org/m"}]),
            teachers_cfg={"teacher_ids": {"math": "math_teacher"}},
            dataset=FakeDataset(
                {"math": (2, 20)},
                teacher_ids={"math": "math_teacher_deepseek_r1_distill_qwen_1p5b"},
            ),
        )
        assert not rep.ok
        assert "math_teacher_deepseek_r1_distill_qwen_1p5b" in " ".join(rep.errors)

    def test_a_matching_teacher_id_passes(self):
        rep = validate(
            data_cfg=_data([{"name": "math", "layer_c": "org/m"}]),
            teachers_cfg={"teacher_ids": {"math": "real_id"}},
            dataset=FakeDataset({"math": (2, 20)}, teacher_ids={"math": "real_id"}),
        )
        assert rep.ok, rep.render()

    def test_a_missing_entry_is_a_warning_not_an_error(self):
        rep = validate(
            data_cfg=_data([{"name": "math", "layer_c": "org/m"}]),
            teachers_cfg={"teacher_ids": {}},
            dataset=FakeDataset({"math": (2, 20)}),
        )
        assert rep.ok
        assert any("teacher_ids" in w for w in rep.warnings)


class TestLiveTeachers:
    def test_an_undeclared_live_teacher_is_an_error(self):
        """It would silently distil that domain through teacher index 0."""
        rep = validate(
            data_cfg=_data([
                {"name": "code", "layer_c": "org/c"},
                {"name": "medical", "layer_c": "org/m"},
            ]),
            teachers_cfg={"mode": "live", "pretrained": {"code": {"name": "X"}}},
        )
        assert not rep.ok
        assert "medical" in " ".join(rep.errors)

    def test_cache_mode_does_not_require_pretrained(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            teachers_cfg={"mode": "cache"},
        )
        assert rep.ok, rep.render()


class TestExactMatchProbes:
    def test_no_gold_rows_is_an_error(self):
        rep = validate(
            data_cfg=_data([{"name": "general", "layer_c": "org/g"}]),
            eval_cfg={"exact_match": {"enabled": True, "domains": ["general"],
                                      "num_samples": 64}},
            dataset=FakeDataset({"general": (4, 100)}, gold={"general": 0}),
        )
        assert not rep.ok
        assert "no has_gold_label rows" in " ".join(rep.errors)

    def test_too_few_gold_rows_is_a_warning_with_the_count(self):
        """math had 3 probes for num_samples=64 and reported nothing."""
        rep = validate(
            data_cfg=_data([{"name": "math", "layer_c": "org/m"}]),
            eval_cfg={"exact_match": {"enabled": True, "domains": ["math"],
                                      "num_samples": 64}},
            dataset=FakeDataset({"math": (4, 100)}, gold={"math": 3}),
        )
        assert rep.ok
        assert "quantized to 1/3" in " ".join(rep.warnings)


class TestNovelRowBudget:
    def test_over_claiming_rows_is_an_error(self):
        rep = validate(
            data_cfg=_data([{"name": "general", "layer_c": "org/g"}]),
            stream_cfg={"phases": [
                {"name": "novel", "domain": "general", "steps": 500, "mode": "novel"},
            ]},
            dataset=FakeDataset({"general": (4, 100)}),  # 400 rows, needs 1000
        )
        assert not rep.ok
        assert "single-use rows" in " ".join(rep.errors)

    def test_a_comfortable_budget_passes(self):
        rep = validate(
            data_cfg=_data([{"name": "general", "layer_c": "org/g"}]),
            stream_cfg={"phases": [
                {"name": "novel", "domain": "general", "steps": 10, "mode": "novel"},
            ]},
            dataset=FakeDataset({"general": (40, 100)}),
        )
        assert rep.ok, rep.render()


class TestMiscellany:
    def test_a_domain_that_failed_to_load_is_an_error(self):
        rep = validate(
            data_cfg=_data([
                {"name": "code", "layer_c": "org/c"},
                {"name": "medical", "layer_c": "org/m"},
            ]),
            dataset=FakeDataset({"code": (4, 100)}),  # medical missing
        )
        assert not rep.ok
        assert "medical" in " ".join(rep.errors)

    def test_mixed_weights_length_mismatch(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [{
                "name": "tail", "domains": ["code"], "steps": 5,
                "mode": "mixed", "weights": [0.5, 0.5],
            }]},
            dataset=FakeDataset({"code": (4, 100)}),
        )
        assert not rep.ok
        assert "weights" in " ".join(rep.errors)

    def test_empty_domains_is_an_error(self):
        rep = validate(data_cfg=_data([]))
        assert not rep.ok

    def test_metadata_only_skips_data_checks(self):
        rep = validate(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            stream_cfg={"phases": [_recurrent("code_recurrent", "code")]},
            dataset=None,
        )
        assert rep.ok
        assert rep.facts["mode"] == "metadata-only"

    def test_validate_or_raise_raises_with_the_full_report(self):
        with pytest.raises(PreflightError, match="needs 8 buckets"):
            validate_or_raise(
                data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
                stream_cfg={"phases": [_recurrent("code_recurrent", "code")]},
                dataset=FakeDataset({"code": (3, 500)}),
            )

    def test_validate_or_raise_returns_the_report_when_clean(self):
        rep = validate_or_raise(
            data_cfg=_data([{"name": "code", "layer_c": "org/c"}]),
            dataset=FakeDataset({"code": (4, 100)}),
        )
        assert rep.ok

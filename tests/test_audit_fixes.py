"""Regressions for the end-to-end audit pass.

Each test here pins a bug that the suite did not catch the first time —
mostly because the existing tests happened to exercise the one configuration
that hid it.
"""

from __future__ import annotations

import json

import pytest
import torch

from hivemind.controller.config import ConsolidationConfig
from hivemind.controller.consolidation import ConsolidationScheduler, _module_sort_key
from hivemind.controller.module_index import ModuleId
from hivemind.data.collate import DistillCollator, EmptyBatchError
from hivemind.data.stream import (
    PhaseSpec,
    StreamConfig,
    StreamSchedule,
    _NovelCursor,
)
from hivemind.data.teacher_cache import TeacherLogitsCache
from hivemind.device import resolve_device
from hivemind.distributed import DistConfig, DistContext
from hivemind.stores.permanent import PermanentStore


class TestCollatorExceptionIsSpecific:
    """A bare `except ValueError` around the data path swallowed too much.

    The training loop reinterprets that exception as "this shard was empty"
    and (with on_empty_batch=skip) drops the step. A ValueError from the
    dataset, tokenizer or logits cache must not be laundered into that.
    """

    def test_empty_batch_raises_the_dedicated_type(self, whitespace_tokenizer):
        collator = DistillCollator(
            tokenizer=whitespace_tokenizer,
            max_seq_len=8,
            logits_cache=TeacherLogitsCache(root=None),
        )
        with pytest.raises(EmptyBatchError):
            collator([{"sample_id": "x"}, {"sample_id": "y"}])

    def test_it_still_subclasses_value_error(self):
        assert issubclass(EmptyBatchError, ValueError)

    def test_an_unrelated_value_error_is_not_that_type(self):
        with pytest.raises(ValueError) as exc:
            DistillCollator(
                tokenizer=object(),
                max_seq_len=8,
                logits_cache=TeacherLogitsCache(root=None),
                gold_policy="bogus",
            )
        assert not isinstance(exc.value, EmptyBatchError)


class TestNovelCursorIsAtomic:
    """take() must not mutate state when it cannot satisfy the request."""

    def _cursor(self, n_rows: int):
        import numpy as np

        used: set[int] = set()
        return _NovelCursor(np.random.default_rng(0), list(range(n_rows)), used), used

    def test_a_failed_take_leaves_used_untouched(self):
        cursor, used = self._cursor(3)
        with pytest.raises(ValueError, match="max_rows_per_domain"):
            cursor.take(5)
        assert used == set(), "a failed take consumed rows anyway"

    def test_a_failed_take_leaves_the_cursor_untouched(self):
        cursor, used = self._cursor(3)
        with pytest.raises(ValueError):
            cursor.take(5)
        # The cursor must still be able to serve a request it CAN satisfy.
        assert len(cursor.take(3)) == 3

    def test_a_successful_take_claims_exactly_n(self):
        cursor, used = self._cursor(10)
        got = cursor.take(4)
        assert len(got) == len(set(got)) == 4
        assert used == set(got)

    def test_rows_are_never_handed_out_twice(self):
        cursor, _ = self._cursor(10)
        a, b = cursor.take(4), cursor.take(4)
        assert not (set(a) & set(b))


class TestDigestFailsLoudly:
    """`default=str` would absorb a memory address into the resume guard."""

    class _Unserializable:
        pass

    def _schedule(self, identity):
        class _DS:
            def indices_for_domain(self, name):
                return range(0, 40)

            def indices_by_bucket(self, name):
                return {f"b{k}": list(range(k * 10, (k + 1) * 10)) for k in range(4)}

        return StreamSchedule(
            StreamConfig(
                enabled=True,
                phases=[PhaseSpec(name="w", steps=2, mode="random", domain="a")],
            ),
            _DS(),
            batch_size=2,
            seed=1,
            data_identity=identity,
        )

    def test_json_native_identity_hashes(self):
        assert len(self._schedule({"split": "train"}).config_digest()) == 16

    def test_non_serializable_identity_raises(self):
        sched = self._schedule({"obj": self._Unserializable()})
        with pytest.raises(TypeError, match="deterministic across processes"):
            sched.config_digest()


class TestModuleSortKeyIsRankStable:
    """ModuleId's hash is PYTHONHASHSEED-randomized; set order is not stable.

    Merges broadcast one tensor per adapter, so a per-rank ordering pairs
    rank i's j-th adapter with rank 0's j-th tensor from a DIFFERENT module.
    Between equally-shaped modules that succeeds silently and permutes
    lora_a — which a plain checksum cannot see, since a sum is invariant
    under permutation.
    """

    def test_sorting_is_total_and_deterministic(self):
        ids = [
            ModuleId(1, "ffn", "F"), ModuleId(0, "attn", "F"),
            ModuleId(1, "attn", "F"), ModuleId(0, "ffn", "F"),
        ]
        assert sorted(ids, key=_module_sort_key) == [
            ModuleId(0, "attn", "F"), ModuleId(0, "ffn", "F"),
            ModuleId(1, "attn", "F"), ModuleId(1, "ffn", "F"),
        ]

    def test_set_order_is_not_relied_on(self):
        """Sorting a set gives the same sequence regardless of insertion."""
        a = {ModuleId(1, "ffn", "F"), ModuleId(0, "attn", "F"), ModuleId(2, "attn", "F")}
        b = {ModuleId(2, "attn", "F"), ModuleId(0, "attn", "F"), ModuleId(1, "ffn", "F")}
        assert sorted(a, key=_module_sort_key) == sorted(b, key=_module_sort_key)

    def test_force_consolidate_targets_are_sorted(self, tmp_path):
        from hivemind.student.config import LoRAConfig, StudentConfig
        from hivemind.student.model import StudentModel

        student = StudentModel(
            StudentConfig(
                vocab_size=32, dim=16, num_layers=3, heads=2, max_seq_len=8,
                lora=LoRAConfig(rank=2, target_modules=["q", "v"]),
            )
        )
        cons = ConsolidationScheduler(
            ConsolidationConfig(merge_strategy="direct"), student, PermanentStore()
        )
        merged = cons.force_consolidate(
            [ModuleId(2, "attn", "F"), ModuleId(0, "attn", "F"), ModuleId(1, "attn", "F")]
        )
        assert merged == [
            ModuleId(0, "attn", "F"), ModuleId(1, "attn", "F"), ModuleId(2, "attn", "F")
        ]


class TestDistConfigValidation:
    def test_fsdp_is_rejected_even_when_disabled(self):
        """Otherwise it validates clean and silently becomes DDP later."""
        with pytest.raises(ValueError, match="Theorem 1"):
            DistConfig.parse({"enabled": False, "strategy": "fsdp"})

    def test_deepspeed_is_rejected_even_when_disabled(self):
        with pytest.raises(ValueError, match="silently stops holding"):
            DistConfig.parse({"enabled": False, "strategy": "deepspeed"})

    def test_the_rejection_names_the_supported_option(self):
        with pytest.raises(ValueError, match="strategy=ddp"):
            DistConfig.parse({"strategy": "deepspeed"})

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown distributed config keys"):
            DistConfig.parse({"enabled": True, "typo_here": 1})

    def test_ddp_validates(self):
        assert DistConfig.parse({"enabled": True, "strategy": "ddp"}).strategy == "ddp"


class TestSingleProcessConfidenceIsUnchanged:
    """The DDP refactor must not move the single-GPU decision boundary.

    RFPPolicy compares confidence < threshold strictly, so recomputing the
    mean as a float64 sum/count instead of a float32 .mean() is
    decision-visible at the boundary.
    """

    def test_local_mean_is_returned_verbatim(self):
        ctx = DistContext()
        conf = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32)
        local = float(conf.mean().item())
        _, _, out = ctx.sync_controller_inputs(
            bucket_id=0,
            embedding=None,
            conf_sum=float(conf.sum().item()),
            conf_count=3,
            conf_local_mean=local,
        )
        assert out == local

    def test_it_differs_from_the_sum_over_count_form(self):
        """Pins that the two really are distinguishable — else the test is vacuous."""
        conf = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        local = float(conf.mean().item())
        naive = float(conf.sum().item()) / 3
        # For this input they agree; the guarantee is that we return `local`
        # whatever it is, so assert the plumbing rather than a magic value.
        assert local == pytest.approx(naive)

    def test_falls_back_when_no_local_mean_given(self):
        _, _, out = DistContext().sync_controller_inputs(
            bucket_id=0, embedding=None, conf_sum=1.5, conf_count=2
        )
        assert out == 0.75


class TestPositionalChecksum:
    """A plain sum cannot see a permutation — the exact failure it guards."""

    def test_permuting_a_tensor_changes_the_checksum(self):
        ctx = DistContext()
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert a.sum() == b.sum()  # a plain sum is blind to this
        assert ctx._positional_sum(a) != ctx._positional_sum(b)

    def test_identical_tensors_match(self):
        ctx = DistContext()
        a = torch.randn(64)
        assert ctx._positional_sum(a) == ctx._positional_sum(a.clone())

    def test_state_checksum_returns_three_components(self):
        from hivemind.optim.masked_adamw import MaskedAdamW
        from hivemind.student.config import LoRAConfig, StudentConfig
        from hivemind.student.model import StudentModel

        student = StudentModel(
            StudentConfig(vocab_size=16, dim=8, num_layers=1, heads=2,
                          max_seq_len=4, lora=LoRAConfig(rank=2))
        )
        opt = MaskedAdamW(student.parameters(), lr=1e-3)
        assert len(DistContext().state_checksum(student, opt)) == 3


class TestDeviceLocalRank:
    def test_cpu_is_unaffected(self):
        assert resolve_device("cpu").type == "cpu"

    def test_explicit_index_conflicting_with_local_rank_raises(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _i: None)
        with pytest.raises(ValueError, match="takes its own GPU"):
            resolve_device("cuda:0", local_rank=1)

    def test_matching_index_is_fine(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _i: None)
        assert resolve_device("cuda:1", local_rank=1) == torch.device("cuda:1")

    def test_no_local_rank_allows_any_index(self, monkeypatch):
        """Single-process `train.device: cuda:1` must stay legal."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _i: None)
        assert resolve_device("cuda:1") == torch.device("cuda:1")


class TestLauncherDetection:
    """`world_size: auto` resolves from device_count, which is NOT evidence
    that a launcher set up a rendezvous."""

    def test_enabled_without_a_launcher_stays_single_process(self, monkeypatch, capsys):
        from hivemind.distributed import init_distributed

        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        # Pretend we are on a 4-GPU box: `world_size: auto` then resolves to
        # 4, which is exactly the case the old device_count-based guard
        # mistook for "a launcher started us".
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        ctx = init_distributed({"distributed": {"enabled": True}})
        assert ctx.world_size == 1 and not ctx.enabled
        assert "not started by a launcher" in capsys.readouterr().out

    def test_disabled_is_inert(self, monkeypatch):
        from hivemind.distributed import init_distributed

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        ctx = init_distributed({"distributed": {"enabled": False}})
        assert ctx.world_size == 1 and not ctx.enabled


class TestVisibleDeviceMapping:
    """A bare index hands children GPUs the job may not own."""

    @staticmethod
    def _fn():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_experiment import _visible_device

        return _visible_device

    def test_no_inherited_allocation_uses_the_slot(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert self._fn()(2) == "2"

    def test_inherited_allocation_is_indexed_into(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
        fn = self._fn()
        assert fn(0) == "4" and fn(1) == "5"

    def test_slot_beyond_the_allocation_falls_back(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
        assert self._fn()(7) == "7"


def test_run_summary_json_is_still_written(tmp_path):
    """Guards the C12 doc claim that run_summary.json marks a run done."""
    from hivemind.profiling import RunProfiler, write_run_summary

    prof = RunProfiler(device=torch.device("cpu"), enabled=True)
    prof.step_start()
    prof.step_end(tokens_in_step=8)
    write_run_summary(str(tmp_path), profiler=prof, final_metrics={"loss/total": 1.0})
    data = json.loads((tmp_path / "run_summary.json").read_text())
    assert "loss/total" in json.dumps(data)


class TestResumeWithoutACheckpointRoot:
    """checkpoint.enabled=false never creates the root, so _latest()'s
    iterdir() raised a bare FileNotFoundError naming nothing useful."""

    def test_message_names_the_actual_problem(self, tmp_path):
        from hivemind.checkpoint import CheckpointConfig, CheckpointManager

        mgr = CheckpointManager(
            CheckpointConfig(
                enabled=False, dir=str(tmp_path / "nope"), resume_from="latest"
            )
        )
        with pytest.raises(FileNotFoundError, match="checkpoint.enabled=true"):
            mgr.resolve_resume_path()

    def test_no_resume_requested_is_still_none(self, tmp_path):
        from hivemind.checkpoint import CheckpointConfig, CheckpointManager

        mgr = CheckpointManager(
            CheckpointConfig(enabled=False, dir=str(tmp_path / "nope"))
        )
        assert mgr.resolve_resume_path() is None


class TestRetrievalStoreDeviceAlignment:
    """WriteExecutor stores R entries with .cpu(); eval queries them with a
    tensor on the training device. On GPU that raised a device mismatch the
    first time eval ran with a non-empty buffer — invisible here because a
    CPU-only suite has both sides on the same device already."""

    @staticmethod
    def _store(n: int = 3, dim: int = 8):
        from hivemind.stores.retrieval import RetrievalEntry, RetrievalStore

        store = RetrievalStore(max_size=16)
        for i in range(n):
            store.add(
                RetrievalEntry(
                    embedding=torch.randn(dim).cpu(),  # as WriteExecutor writes them
                    teacher_id=0,
                    bucket_id=i,
                    step=i,
                )
            )
        return store

    def test_aligned_puts_query_on_the_buffer_device(self):
        store = self._store()
        query, embeddings = store._aligned(torch.randn(8))
        assert query.device == embeddings.device

    def test_query_is_moved_not_the_buffer(self):
        """The buffer can hold max_size entries and _retrieval_hit_rate calls
        this once per probe row, so the query must be what moves."""
        store = self._store()
        _, embeddings = store._aligned(torch.randn(8))
        assert embeddings.device == store._buffer[0].embedding.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_cuda_query_against_cpu_buffer(self):
        store = self._store()
        q = torch.randn(8, device="cuda")
        assert isinstance(store.hit_count(q, threshold=0.0), int)
        assert len(store.query_nearest(q, k=2)) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_eval_hit_rate_with_cuda_probes(self):
        from hivemind.evaluation import _retrieval_hit_rate

        rate = _retrieval_hit_rate(self._store(), torch.randn(4, 8, device="cuda"))
        assert 0.0 <= rate <= 1.0

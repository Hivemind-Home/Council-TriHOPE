"""Tests for the experiment runner (matrix expansion + status; no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_experiment import (  # noqa: E402
    ResolvedRun,
    cleanup_checkpoints,
    expand_matrix,
    load_manifest,
    main,
)


def _manifest(tmp_path: Path, **kwargs) -> Path:
    payload = {
        "experiment": "exp_v1",
        "base_config": "stream_small",
        "output_root": str(tmp_path / "runs"),
        "seeds": [1, 2],
        "runs": [
            {"id": "trihope", "overrides": []},
            {"id": "no_surprise", "overrides": ["controller.ablation.disable_signals=[surprise]"]},
        ],
    }
    payload.update(kwargs)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


class TestExpandMatrix:
    def test_runs_times_seeds(self, tmp_path: Path) -> None:
        manifest = load_manifest(_manifest(tmp_path))
        runs = expand_matrix(manifest)
        assert [r.run_id for r in runs] == [
            "trihope-seed1", "trihope-seed2",
            "no_surprise-seed1", "no_surprise-seed2",
        ]
        r = runs[2]
        assert r.spec_id == "no_surprise"
        assert r.seed == 1
        assert "controller.ablation.disable_signals=[surprise]" in r.overrides
        assert "++train.seed=1" in r.overrides
        assert any(o.startswith("++checkpoint.dir=") for o in r.overrides)
        assert any(o.startswith("++logging.events_path=") for o in r.overrides)
        assert any(o.startswith("++run.dir=") for o in r.overrides)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        path = _manifest(
            tmp_path, runs=[{"id": "a", "overrides": []}, {"id": "a", "overrides": []}]
        )
        with pytest.raises(ValueError, match="Duplicate run id"):
            expand_matrix(load_manifest(path))

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"experiment": "x"}))
        with pytest.raises(ValueError, match="base_config"):
            load_manifest(path)


class TestDryRun:
    def test_dry_run_prints_matrix(self, tmp_path: Path, capsys) -> None:
        path = _manifest(tmp_path)
        assert main([str(path), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "trihope-seed1" in out and "no_surprise-seed2" in out

    def test_only_filters(self, tmp_path: Path, capsys) -> None:
        path = _manifest(tmp_path)
        assert main([str(path), "--dry-run", "--only", "trihope"]) == 0
        out = capsys.readouterr().out
        assert "trihope-seed1" in out and "no_surprise" not in out

    def test_only_no_match(self, tmp_path: Path) -> None:
        assert main([str(_manifest(tmp_path)), "--dry-run", "--only", "bogus"]) == 2


class TestCleanup:
    def _run_with_ckpts(self, tmp_path: Path) -> ResolvedRun:
        run = ResolvedRun(
            run_id="r-seed1", spec_id="r", seed=1, run_dir=tmp_path / "r-seed1"
        )
        ckpt = run.run_dir / "checkpoints"
        for name in ("step_00000010", "step_00000020", "step_00000015_pre_merge"):
            (ckpt / name).mkdir(parents=True)
        return run

    def test_keep_final(self, tmp_path: Path) -> None:
        run = self._run_with_ckpts(tmp_path)
        cleanup_checkpoints(run, "keep_final")
        remaining = sorted(p.name for p in (run.run_dir / "checkpoints").iterdir())
        # newest untagged + all tagged survive
        assert remaining == ["step_00000015_pre_merge", "step_00000020"]

    def test_keep_none(self, tmp_path: Path) -> None:
        run = self._run_with_ckpts(tmp_path)
        cleanup_checkpoints(run, "keep_none")
        assert list((run.run_dir / "checkpoints").iterdir()) == []

    def test_all_keeps_everything(self, tmp_path: Path) -> None:
        run = self._run_with_ckpts(tmp_path)
        cleanup_checkpoints(run, "all")
        assert len(list((run.run_dir / "checkpoints").iterdir())) == 3


class TestManifestFilesParse:
    """The checked-in manifests must always expand cleanly."""

    @pytest.mark.parametrize(
        "name",
        ["baselines_small", "ablation_grid", "p_study_small", "headline"],
    )
    def test_checked_in_manifest(self, name: str) -> None:
        path = Path(__file__).parent.parent / "configs" / "experiments" / f"{name}.yaml"
        manifest = load_manifest(path)
        runs = expand_matrix(manifest)
        assert len(runs) >= 4
        assert len({r.run_id for r in runs}) == len(runs)


class TestLaunchModes:
    """Two ways to spend N GPUs, with different trade-offs."""

    @staticmethod
    def _run(tmp_path):
        from run_experiment import ResolvedRun

        return ResolvedRun(
            run_id="trihope-seed1", spec_id="trihope", seed=1,
            run_dir=tmp_path / "trihope-seed1",
            overrides=["++train.seed=1"],
        )

    def test_single_process_command_is_unchanged(self, tmp_path):
        from run_experiment import build_command

        cmd = build_command(self._run(tmp_path), "stream_small", nproc=1, resume=False)
        assert cmd[1].endswith("train.py")
        assert "torch.distributed.run" not in " ".join(cmd)
        assert "++distributed.enabled=true" not in cmd

    def test_multi_process_uses_torch_distributed_run(self, tmp_path):
        from run_experiment import build_command

        cmd = build_command(self._run(tmp_path), "stream_small", nproc=4, resume=False)
        joined = " ".join(cmd)
        # -m torch.distributed.run, not the `torchrun` shim: guarantees the
        # same interpreter under conda/uv.
        assert cmd[1] == "-m" and cmd[2] == "torch.distributed.run"
        assert "--nproc_per_node=4" in cmd
        assert "++distributed.enabled=true" in cmd
        # Forwarded so init_distributed can cross-check against WORLD_SIZE.
        assert "++distributed.world_size=4" in cmd
        # Per-rank log files, else N ranks interleave into one fd.
        assert "--redirects=3" in cmd and "torchrun_logs" in joined

    def test_resume_flag_is_appended_last_in_both_modes(self, tmp_path):
        from run_experiment import build_command

        run = self._run(tmp_path)
        (run.run_dir / "checkpoints").mkdir(parents=True)
        for nproc in (1, 4):
            cmd = build_command(run, "stream_small", nproc=nproc, resume=True)
            assert cmd[-1] == "++checkpoint.resume_from=latest"

    def test_hydra_run_dir_is_per_rank(self):
        from run_experiment import expand_matrix

        runs = expand_matrix(
            {"experiment": "e", "base_config": "stream_small",
             "runs": [{"id": "a"}], "seeds": [1]}
        )
        hydra = [o for o in runs[0].overrides if o.startswith("++hydra.run.dir=")]
        assert len(hydra) == 1
        # N ranks writing one .hydra/ snapshot concurrently is a race.
        assert "${oc.env:RANK,0}" in hydra[0]

    def test_the_two_gpu_modes_are_mutually_exclusive(self, tmp_path, capsys):
        import pytest as _pytest
        from run_experiment import main

        manifest = tmp_path / "m.yaml"
        manifest.write_text(
            "experiment: e\nbase_config: stream_small\nruns:\n  - id: a\n"
        )
        with _pytest.raises(SystemExit):
            main([str(manifest), "--parallel-gpus", "2", "--nproc-per-node", "2"])
        assert "pick one" in capsys.readouterr().err

"""Tests for the experiment runner (matrix expansion + status; no training)."""

from __future__ import annotations

import json
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
        assert f"++train.seed=1" in r.overrides
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

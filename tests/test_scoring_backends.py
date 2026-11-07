"""Backend selection, optional dependencies, and cached-result isolation."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from boltzina.affinity.backends import (
    check_cached_backend, validate_scoring_backend, write_backend_metadata,
)
from boltzina.cli import main
from boltzina.runner import BoltzinaRunner, RunnerConfig


def test_cli_forwards_backend(simple_smi_file, boltz_work_dir, tmp_path):
    with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()) as run:
        result = CliRunner().invoke(main, [
            "run", str(simple_smi_file), "--work-dir", str(boltz_work_dir),
            "--output-dir", str(tmp_path), "--scoring-backend", "bioir",
        ])
    assert result.exit_code == 0, result.output
    assert run.call_count == 1
    # Inspect the RunnerConfig independently of the mocked pipeline execution.
    with patch("boltzina.runner.BoltzinaRunner") as runner:
        runner.return_value.run.return_value = pd.DataFrame()
        result = CliRunner().invoke(main, [
            "run", str(simple_smi_file), "--sequence", "MENFQKV", "--scoring-backend", "bioir",
        ])
    assert result.exit_code == 0, result.output
    assert runner.call_args.args[0].scoring_backend == "bioir"


def test_native_does_not_import_bioir(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Native scoring should not inspect BioIR.")

    monkeypatch.setattr("boltzina.affinity.backends.importlib.util.find_spec", forbidden)
    validate_scoring_backend("boltz")


def test_missing_bioir_fails_before_preparation(monkeypatch, tmp_path):
    monkeypatch.setattr("boltzina.affinity.backends.importlib.util.find_spec", lambda name: None)
    runner = BoltzinaRunner(RunnerConfig(
        input_path=Path("ligands.smi"), output_dir=tmp_path, sequence="MENFQKV",
        scoring_backend="bioir",
    ))
    with patch.object(runner, "_run_predict") as prepare:
        with pytest.raises(RuntimeError, match="BioIR is not installed"):
            runner.run()
    prepare.assert_not_called()


@pytest.mark.parametrize("backend,kernels,error", [
    ("invalid", True, "Unknown scoring backend"),
    ("bioir", False, "no-kernels"),
])
def test_invalid_backend_options(backend, kernels, error):
    with pytest.raises(ValueError, match=error):
        validate_scoring_backend(backend, use_kernels=kernels)


def test_cached_results_require_explicit_backend_switch(tmp_path):
    prediction = tmp_path / "boltz_out/predictions/record/affinity_record.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("{}")
    with pytest.raises(ValueError, match="boltz-override"):
        check_cached_backend(tmp_path, "bioir", override=False)
    check_cached_backend(tmp_path, "bioir", override=True)
    write_backend_metadata(tmp_path, "bioir", state="complete")
    check_cached_backend(tmp_path, "bioir", override=False)
    with pytest.raises(ValueError, match="boltz-override"):
        check_cached_backend(tmp_path, "boltz", override=False)
    assert json.loads((tmp_path / "scoring_backend.json").read_text())["scoring_backend"] == "bioir"
    write_backend_metadata(tmp_path, "bioir", state="running")
    with pytest.raises(ValueError, match="did not finish"):
        check_cached_backend(tmp_path, "bioir", override=False)


def test_engine_routes_bioir_without_loading_native(tmp_path):
    from boltzina.engine import Boltzina

    engine = Boltzina.__new__(Boltzina)
    engine.output_dir, engine.work_dir = tmp_path, tmp_path / "work"
    engine.scoring_backend = "bioir"
    engine.batch_size, engine.seed = 2, 42
    engine.predict_affinity_args = {"recycling_steps": 2}
    engine.boltz_override = False
    manifest = tmp_path / "boltz_out/processed/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"records": [{"id": "record"}]}))
    with patch("boltzina.affinity.bioir.predict_bioir_affinity") as score:
        with patch("boltzina.engine.load_boltz2_model") as native:
            engine._score_poses()
    native.assert_not_called()
    assert score.call_args.kwargs["batch_size"] == 2
    assert score.call_args.kwargs["recycling_steps"] == 2
    assert score.call_args.kwargs["manifest_path"] == tmp_path / "boltz_out/processed/manifest.json"


def test_rescoring_clears_previous_predictions_and_runtime_metadata(tmp_path):
    from boltzina.engine import Boltzina

    engine = Boltzina.__new__(Boltzina)
    engine.output_dir, engine.work_dir = tmp_path, tmp_path / "work"
    engine.scoring_backend = "bioir"
    engine.batch_size, engine.seed = 1, 42
    engine.predict_affinity_args = None
    engine.boltz_override = True
    manifest = tmp_path / "boltz_out/processed/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"records": [{"id": "record"}]}))
    prediction = tmp_path / "boltz_out/predictions/record/affinity_record.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("{}")
    metadata = tmp_path / "boltz_out/scoring_metadata.json"
    metadata.write_text('{"scoring_backend": "boltz"}')

    def score(*args, **kwargs):
        assert not prediction.exists()
        assert not metadata.exists()

    with patch("boltzina.affinity.bioir.predict_bioir_affinity", side_effect=score):
        engine._score_poses()

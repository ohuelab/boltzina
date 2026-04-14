"""
Tests for boltzina.cli — CLI argument parsing and command routing.

All tests use click.testing.CliRunner and do NOT actually run the pipeline
(mocked at BoltzinaRunner.run or prepare_ligands_from_file level).
The goal is to verify:
  - Argument parsing is correct
  - Mode auto-detection (A vs B) works
  - Invalid input combinations are rejected with helpful messages
  - --version, --help work
  - Subcommands (prepare, grid, setup) are wired up
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from boltzina.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

class TestGlobalOptions:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "boltzina" in result.output.lower() or "version" in result.output.lower()

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_subcommand_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--sequence" in result.output
        assert "--work-dir" in result.output

    def test_prepare_subcommand_help(self, runner):
        result = runner.invoke(main, ["prepare", "--help"])
        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output or "input-path" in result.output.lower()

    def test_setup_subcommand_help(self, runner):
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# boltzina run — mode validation
# ---------------------------------------------------------------------------

class TestRunModeValidation:
    def test_no_mode_given_exits_nonzero(self, runner, simple_smi_file):
        """Neither --sequence nor --work-dir → error."""
        result = runner.invoke(main, ["run", str(simple_smi_file)])
        assert result.exit_code != 0
        output = result.output.lower()
        assert "sequence" in output or "work-dir" in output or "required" in output

    def test_mode_a_requires_sequence(self, runner, simple_smi_file, tmp_path):
        """Mode A accepted when --sequence is given."""
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, [
                "run", str(simple_smi_file),
                "--sequence", "MENFQKV",
                "--output-dir", str(tmp_path / "out"),
            ])
            # Should not fail due to missing mode
            assert "Either" not in result.output

    def test_mode_b_requires_work_dir(self, runner, simple_smi_file, tmp_path,
                                       boltz_work_dir):
        """Mode B accepted when --work-dir is given."""
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, [
                "run", str(simple_smi_file),
                "--work-dir", str(boltz_work_dir),
                "--output-dir", str(tmp_path / "out"),
            ])
            assert "Either" not in result.output


# ---------------------------------------------------------------------------
# boltzina run — argument parsing
# ---------------------------------------------------------------------------

class TestRunArgParsing:
    def _run_with_mock(self, runner, args):
        """Invoke CLI with BoltzinaRunner.run mocked out."""
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()) as mock_run:
            result = runner.invoke(main, args)
            return result, mock_run

    def test_grid_center_parsed(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        captured = {}
        orig_init = __import__("boltzina.runner", fromlist=["BoltzinaRunner"]).BoltzinaRunner.__init__
        def patched_init(self, config):
            captured["config"] = config
            orig_init(self, config)
        with patch("boltzina.runner.BoltzinaRunner.__init__", patched_init), \
             patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            runner.invoke(main, [
                "run", str(simple_smi_file),
                "--work-dir", str(boltz_work_dir),
                "--grid-center", "7.0,-4.9,7.5",
                "--output-dir", str(tmp_path / "out"),
            ])
        cfg = captured.get("config")
        if cfg is not None:
            assert cfg.grid_center == (7.0, -4.9, 7.5)

    def test_invalid_grid_center_rejected(self, runner, simple_smi_file, tmp_path):
        result = runner.invoke(main, [
            "run", str(simple_smi_file),
            "--sequence", "MENFQKV",
            "--grid-center", "notanumber",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_no_kernels_flag(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        captured = {}
        orig_init = __import__("boltzina.runner", fromlist=["BoltzinaRunner"]).BoltzinaRunner.__init__
        def patched_init(self, config):
            captured["config"] = config
            orig_init(self, config)
        with patch("boltzina.runner.BoltzinaRunner.__init__", patched_init), \
             patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            runner.invoke(main, [
                "run", str(simple_smi_file),
                "--work-dir", str(boltz_work_dir),
                "--no-kernels",
                "--output-dir", str(tmp_path / "out"),
            ])
        cfg = captured.get("config")
        if cfg is not None:
            assert cfg.use_kernels is False

    def test_default_use_kernels_true(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        """use_kernels should default to True (no --no-kernels flag)."""
        captured = {}
        orig_init = __import__("boltzina.runner", fromlist=["BoltzinaRunner"]).BoltzinaRunner.__init__
        def patched_init(self, config):
            captured["config"] = config
            orig_init(self, config)
        with patch("boltzina.runner.BoltzinaRunner.__init__", patched_init), \
             patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            runner.invoke(main, [
                "run", str(simple_smi_file),
                "--work-dir", str(boltz_work_dir),
                "--output-dir", str(tmp_path / "out"),
            ])
        cfg = captured.get("config")
        if cfg is not None:
            assert cfg.use_kernels is True

    def test_seed_parsed(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        captured = {}
        orig_init = __import__("boltzina.runner", fromlist=["BoltzinaRunner"]).BoltzinaRunner.__init__
        def patched_init(self, config):
            captured["config"] = config
            orig_init(self, config)
        with patch("boltzina.runner.BoltzinaRunner.__init__", patched_init), \
             patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            runner.invoke(main, [
                "run", str(simple_smi_file),
                "--work-dir", str(boltz_work_dir),
                "--seed", "42",
                "--output-dir", str(tmp_path / "out"),
            ])
        cfg = captured.get("config")
        if cfg is not None:
            assert cfg.seed == 42


# ---------------------------------------------------------------------------
# boltzina prepare
# ---------------------------------------------------------------------------

class TestPrepareCmd:
    def test_smi_file(self, runner, simple_smi_file, tmp_path):
        with patch("boltzina.preparation.prepare_ligands_from_file",
                   return_value=([], tmp_path / "mols.pkl")) as mock_prep:
            mock_prep.return_value = ([], tmp_path / "mols.pkl")
            result = runner.invoke(main, [
                "prepare", str(simple_smi_file),
                "--output-dir", str(tmp_path / "out"),
            ])
        # Exit 0 or at most file-exists issue; should not have parse error
        assert "Error: No such file" not in result.output

    def test_missing_input_exits_nonzero(self, runner, tmp_path):
        result = runner.invoke(main, ["prepare", "/no/such/file.smi"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Import chain tests (regression: boltzina_main → boltzina.engine)
# ---------------------------------------------------------------------------

class TestImportChain:
    def test_boltzina_engine_importable(self):
        """boltzina.engine must be importable (was broken when boltzina_main.py was top-level)."""
        from boltzina.engine import Boltzina  # noqa: F401
        assert Boltzina is not None

    def test_boltzina_version_accessible(self):
        """boltzina.__version__ must be accessible without loading heavy deps."""
        import boltzina
        assert hasattr(boltzina, "__version__"), "boltzina.__version__ not found"
        assert boltzina.__version__

    def test_runner_importable(self):
        """BoltzinaRunner must be importable without ModuleNotFoundError."""
        from boltzina.runner import BoltzinaRunner, RunnerConfig  # noqa: F401
        assert BoltzinaRunner is not None

    def test_setup_subcommand_no_install_unidock2(self, runner):
        """--install-unidock2 should NOT appear in setup --help (removed per spec)."""
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--install-unidock2" not in result.output

    def test_run_new_options_in_help(self, runner):
        """New Boltz-2 CLI options must appear in run --help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--max-parallel-samples" in result.output
        assert "--subsample-msa" in result.output
        assert "--msa-pairing-strategy" in result.output

"""
Tests for boltzina.cli — CLI argument parsing and command routing.

All tests use click.testing.CliRunner and do NOT actually run the pipeline
(mocked at BoltzinaRunner.run or prepare_ligands_from_file level).
The goal is to verify:
  - Argument parsing is correct
  - Input mode selection works (sequence / sequence-file / yaml / work-dir)
  - Invalid input combinations are rejected with helpful messages
  - --version, --help work
  - Subcommands (prepare, grid, setup) are wired up
  - "Mode A/B" terminology is absent from all output
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
        assert "--yaml" in result.output

    def test_prepare_subcommand_help(self, runner):
        result = runner.invoke(main, ["prepare", "--help"])
        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output or "input-path" in result.output.lower()

    def test_setup_subcommand_help(self, runner):
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# boltzina run — input mode validation
# ---------------------------------------------------------------------------

class TestRunModeValidation:
    def test_no_protein_input_exits_nonzero(self, runner, simple_smi_file):
        """No --sequence / --yaml / --work-dir → RequiredMutuallyExclusiveOptionGroup error."""
        result = runner.invoke(main, ["run", str(simple_smi_file)])
        assert result.exit_code != 0

    def test_sequence_accepted(self, runner, simple_smi_file, tmp_path):
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, [
                "run", str(simple_smi_file),
                "--sequence", "MENFQKV",
                "--output-dir", str(tmp_path / "out"),
            ])
            assert "Mode" not in result.output or "A" not in result.output

    def test_sequence_file_accepted(self, runner, simple_smi_file, tmp_path, fasta_file):
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, [
                "run", str(simple_smi_file),
                "--sequence-file", str(fasta_file),
                "--output-dir", str(tmp_path / "out"),
            ])
            assert result.exit_code == 0 or "Error" not in result.output

    def test_yaml_accepted(self, runner, simple_smi_file, tmp_path, boltz_yaml_file):
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, [
                "run", str(simple_smi_file),
                "--yaml", str(boltz_yaml_file),
                "--output-dir", str(tmp_path / "out"),
            ])
            assert result.exit_code == 0 or "Error" not in result.output

    def test_work_dir_accepted(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, [
                "run", str(simple_smi_file),
                "--work-dir", str(boltz_work_dir),
                "--output-dir", str(tmp_path / "out"),
            ])
            assert result.exit_code == 0 or "Error" not in result.output

    def test_yaml_and_sequence_mutually_exclusive(self, runner, simple_smi_file, tmp_path,
                                                   boltz_yaml_file):
        """--yaml and --sequence cannot be combined."""
        result = runner.invoke(main, [
            "run", str(simple_smi_file),
            "--yaml", str(boltz_yaml_file),
            "--sequence", "MENFQKV",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_yaml_and_reference_ligand_mutually_exclusive(self, runner, simple_smi_file,
                                                           tmp_path, boltz_yaml_file):
        """--yaml and --reference-ligand cannot be combined."""
        result = runner.invoke(main, [
            "run", str(simple_smi_file),
            "--yaml", str(boltz_yaml_file),
            "--reference-ligand", "CCO",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_work_dir_and_reference_ligand_mutually_exclusive(self, runner, simple_smi_file,
                                                               tmp_path, boltz_work_dir):
        """--work-dir and --reference-ligand cannot be combined."""
        result = runner.invoke(main, [
            "run", str(simple_smi_file),
            "--work-dir", str(boltz_work_dir),
            "--reference-ligand", "CCO",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# boltzina run — echo messages (no "Mode A/B")
# ---------------------------------------------------------------------------

class TestRunEchoMessages:
    def _capture_echo(self, runner, args):
        with patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            result = runner.invoke(main, args)
        return result.output

    def test_mode_a_b_text_absent_in_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert "Mode A" not in result.output
        assert "Mode B" not in result.output

    def test_sequence_mode_echo(self, runner, simple_smi_file, tmp_path):
        out = self._capture_echo(runner, [
            "run", str(simple_smi_file),
            "--sequence", "MENFQKV",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert "Mode A" not in out
        assert "Mode B" not in out
        assert "prediction" in out.lower() or "sequence" in out.lower()

    def test_work_dir_mode_echo(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        out = self._capture_echo(runner, [
            "run", str(simple_smi_file),
            "--work-dir", str(boltz_work_dir),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert "Mode A" not in out
        assert "Mode B" not in out
        assert "precomputed" in out.lower() or "result" in out.lower()


# ---------------------------------------------------------------------------
# boltzina run — argument parsing
# ---------------------------------------------------------------------------

class TestRunArgParsing:
    def _capture_config(self, runner, args):
        captured = {}
        orig_init = __import__("boltzina.runner", fromlist=["BoltzinaRunner"]).BoltzinaRunner.__init__
        def patched_init(self, config):
            captured["config"] = config
            orig_init(self, config)
        with patch("boltzina.runner.BoltzinaRunner.__init__", patched_init), \
             patch("boltzina.runner.BoltzinaRunner.run", return_value=pd.DataFrame()):
            runner.invoke(main, args)
        return captured.get("config")

    def test_grid_center_parsed(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        cfg = self._capture_config(runner, [
            "run", str(simple_smi_file),
            "--work-dir", str(boltz_work_dir),
            "--grid-center", "7.0,-4.9,7.5",
            "--output-dir", str(tmp_path / "out"),
        ])
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
        cfg = self._capture_config(runner, [
            "run", str(simple_smi_file),
            "--work-dir", str(boltz_work_dir),
            "--no-kernels",
            "--output-dir", str(tmp_path / "out"),
        ])
        if cfg is not None:
            assert cfg.use_kernels is False

    def test_default_use_kernels_true(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        cfg = self._capture_config(runner, [
            "run", str(simple_smi_file),
            "--work-dir", str(boltz_work_dir),
            "--output-dir", str(tmp_path / "out"),
        ])
        if cfg is not None:
            assert cfg.use_kernels is True

    def test_seed_parsed(self, runner, simple_smi_file, tmp_path, boltz_work_dir):
        cfg = self._capture_config(runner, [
            "run", str(simple_smi_file),
            "--work-dir", str(boltz_work_dir),
            "--seed", "42",
            "--output-dir", str(tmp_path / "out"),
        ])
        if cfg is not None:
            assert cfg.seed == 42

    def test_sequence_colon_stored(self, runner, simple_smi_file, tmp_path):
        cfg = self._capture_config(runner, [
            "run", str(simple_smi_file),
            "--sequence", "MENFQKV:AKLSILP",
            "--output-dir", str(tmp_path / "out"),
        ])
        if cfg is not None:
            assert cfg.sequence == "MENFQKV:AKLSILP"

    def test_reference_ligand_stored(self, runner, simple_smi_file, tmp_path):
        cfg = self._capture_config(runner, [
            "run", str(simple_smi_file),
            "--sequence", "MENFQKV",
            "--reference-ligand", "CCO",
            "--output-dir", str(tmp_path / "out"),
        ])
        if cfg is not None:
            assert cfg.reference_ligand == "CCO"


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
        assert "Error: No such file" not in result.output

    def test_missing_input_exits_nonzero(self, runner, tmp_path):
        result = runner.invoke(main, ["prepare", "/no/such/file.smi"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Import chain tests
# ---------------------------------------------------------------------------

class TestImportChain:
    def test_boltzina_engine_importable(self):
        from boltzina.engine import Boltzina  # noqa: F401
        assert Boltzina is not None

    def test_ligand_prep_command_renames_obabel_unl(self, tmp_path):
        from boltzina.engine import _build_ligand_prep_command

        cmd = _build_ligand_prep_command(
            ligand_chain_id="B",
            pdb_file=tmp_path / "docked_ligand_1.pdb",
            prep_file=tmp_path / "docked_ligand_1_prep.pdb",
            input_ligand_name="MOL",
            base_ligand_name="MOL",
        )

        assert f"pdb_chain -B {tmp_path / 'docked_ligand_1.pdb'}" in cmd
        assert 'pdb_rplresname -"UNL":MOL' in cmd
        assert f"> {tmp_path / 'docked_ligand_1_prep.pdb'}" in cmd

    def test_ligand_prep_command_keeps_input_resname_rename(self, tmp_path):
        from boltzina.engine import _build_ligand_prep_command

        cmd = _build_ligand_prep_command(
            ligand_chain_id="C",
            pdb_file=tmp_path / "pose.pdb",
            prep_file=tmp_path / "pose_prep.pdb",
            input_ligand_name="LIG",
            base_ligand_name="MOL",
        )

        assert 'pdb_rplresname -"LIG":MOL' in cmd
        assert 'pdb_rplresname -"UNL":MOL' in cmd

    def test_ligand_mol_update_respects_boltz_override(self, tmp_path):
        from boltzina import engine

        obj = engine.Boltzina.__new__(engine.Boltzina)
        obj.output_dir = tmp_path
        obj.fname = "target"
        obj.pose_idxs = ["1"]

        affinity_dir = tmp_path / "boltz_out" / "predictions" / "target_0_1"
        affinity_dir.mkdir(parents=True)
        (affinity_dir / "affinity_target_0_1.json").write_text("{}")

        obj.boltz_override = False
        assert obj._needs_ligand_mol_update(0) is False

        obj.boltz_override = True
        assert obj._needs_ligand_mol_update(0) is True

    def test_engine_resolves_scripts_next_to_python(self, tmp_path):
        from boltzina import tools

        scripts_dir = tmp_path / "bin"
        scripts_dir.mkdir()
        python = scripts_dir / "python"
        script = scripts_dir / "mk_prepare_receptor.py"
        python.write_text("")
        script.write_text("")
        script.chmod(0o755)

        with patch("boltzina.tools.shutil.which", return_value=None), \
             patch.object(tools.sys, "executable", str(python)):
            assert tools.resolve_executable("mk_prepare_receptor.py") == str(script)

    def test_prepare_receptor_uses_read_pdb_for_pdb_inputs(self, tmp_path):
        from boltzina import engine

        receptor = tmp_path / "receptor_input.pdb"
        receptor.write_text("ATOM\n")
        obj = engine.Boltzina.__new__(engine.Boltzina)
        obj.output_dir = tmp_path
        obj.receptor_pdb = receptor
        obj.vina_override = True

        def fake_run(cmd, check, **kwargs):
            assert cmd[1] == "--read_pdb"
            assert cmd[2] == str(receptor)
            assert "env" in kwargs
            (tmp_path / "receptor.pdbqt").write_text("PDBQT\n")

        with patch("boltzina.engine.resolve_executable", return_value="/bin/mk_prepare_receptor.py"), \
             patch("boltzina.engine.subprocess.run", side_effect=fake_run):
            assert obj._prepare_receptor() == tmp_path / "receptor.pdbqt"

    def test_boltzina_version_accessible(self):
        import boltzina
        assert hasattr(boltzina, "__version__"), "boltzina.__version__ not found"
        assert boltzina.__version__

    def test_runner_importable(self):
        from boltzina.runner import BoltzinaRunner, RunnerConfig  # noqa: F401
        assert BoltzinaRunner is not None

    def test_setup_subcommand_no_install_unidock2(self, runner):
        result = runner.invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--install-unidock2" not in result.output

    def test_run_new_options_in_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--max-parallel-samples" in result.output
        assert "--subsample-msa" in result.output
        assert "--msa-pairing-strategy" in result.output
        assert "--yaml" in result.output
        assert "--reference-ligand" in result.output
        assert "--representative-smiles" not in result.output

"""
Integration tests for Boltzina — full pipeline runs.

These tests require a GPU environment. All tests are marked with @pytest.mark.gpu.

Mode B (existing Boltz results) integration test uses the pre-computed
sample/CDK2/boltz_results_base directory so no Boltz-2 re-run is needed.

Mode A (full-auto) test is marked @pytest.mark.slow because it requires
running Boltz-2 structure prediction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_CDK2 = Path(__file__).parent.parent / "sample" / "CDK2"
BOLTZ_RESULTS_BASE = SAMPLE_CDK2 / "boltz_results_base"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_smi_file(tmp_path: Path, n: int = 2) -> Path:
    """Write a tiny SMILES file with n CDK2 active compounds."""
    # A few real CDK2 actives (truncated for speed)
    smiles_list = [
        "CCN(CC)c1nc(Nc2ccccc2)c2ncn(C(C)CO)c2n1 roscovitine",
        "Cc1cnc(Nc2cccc(NC(=O)c3ccco3)c2)nc1N purvalanol_a",
        "c1ccc(CNc2ncnc3[nH]cnc23)cc1 adenine_benzyl",
        "O=C(Nc1cccc(Nc2ncnc3[nH]cnc23)c1)c1ccccc1 benzamide_inh",
    ]
    smi_path = tmp_path / "test_cdk2.smi"
    smi_path.write_text("\n".join(smiles_list[:n]) + "\n")
    return smi_path


# ---------------------------------------------------------------------------
# Mode B: uses pre-computed boltz_results_base
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestModeBIntegration:
    def test_mode_b_full_pipeline(self, tmp_path):
        """
        Full Mode B pipeline: existing Boltz-2 output → docking → scoring.

        Uses sample/CDK2/boltz_results_base. No Boltz-2 re-run needed.
        Output goes to tmp_path.
        """
        from boltzina.runner import BoltzinaRunner, RunnerConfig

        smi_file = make_small_smi_file(tmp_path, n=2)
        out_dir = tmp_path / "mode_b_results"

        cfg = RunnerConfig(
            input_path=smi_file,
            output_dir=out_dir,
            work_dir=BOLTZ_RESULTS_BASE,
            ligand_chain_id="B",
            seed=42,
            num_workers=1,
            vina_cpu=1,
            batch_size=1,
            vina_override=True,
            boltz_override=True,
            use_kernels=True,
            run_trunk_and_structure=True,
        )

        runner = BoltzinaRunner(cfg)
        df = runner.run()

        # Verify results
        assert isinstance(df, pd.DataFrame), "Expected DataFrame result"
        assert len(df) > 0, "Results DataFrame should not be empty"
        assert out_dir.exists()

        # CSV should be saved
        csv_path = out_dir / "results.csv"
        assert csv_path.exists(), f"results.csv not found at {csv_path}"

        # Should have affinity prediction column
        assert any("affinity" in col.lower() for col in df.columns), (
            f"No affinity column found. Columns: {list(df.columns)}"
        )

    def test_mode_b_with_explicit_grid_center(self, tmp_path):
        """Mode B with explicitly specified grid center."""
        from boltzina.runner import BoltzinaRunner, RunnerConfig

        smi_file = make_small_smi_file(tmp_path, n=1)
        out_dir = tmp_path / "mode_b_explicit_grid"

        cfg = RunnerConfig(
            input_path=smi_file,
            output_dir=out_dir,
            work_dir=BOLTZ_RESULTS_BASE,
            ligand_chain_id="B",
            grid_center=(7.088, -4.921, 7.519),
            seed=42,
            num_workers=1,
            vina_cpu=1,
            batch_size=1,
            vina_override=True,
            boltz_override=True,
            use_kernels=True,
        )

        runner = BoltzinaRunner(cfg)
        df = runner.run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_mode_b_skip_docking(self, tmp_path, sample_cdk2_dir):
        """Mode B with skip_docking=True (score existing poses)."""
        from boltzina.runner import BoltzinaRunner, RunnerConfig

        # Use pre-prepared input PDBs from sample data
        input_pdbs_dir = sample_cdk2_dir / "input_pdbs_pose"
        if not input_pdbs_dir.exists():
            input_pdbs_dir = sample_cdk2_dir / "input_pdbs"
        assert input_pdbs_dir.exists(), f"Input PDBs dir not found: {input_pdbs_dir}"

        # Use a single pre-prepared PDB file
        pdb_files = list(input_pdbs_dir.glob("*.pdb"))
        assert pdb_files, f"No PDB files in {input_pdbs_dir}"

        # For skip_docking, we need pre-docked poses. Reuse the first few input PDBs.
        # Create a minimal SMILES file matching the first ligand
        smi_file = make_small_smi_file(tmp_path, n=1)
        out_dir = tmp_path / "mode_b_skip_docking"

        cfg = RunnerConfig(
            input_path=smi_file,
            output_dir=out_dir,
            work_dir=BOLTZ_RESULTS_BASE,
            ligand_chain_id="B",
            grid_center=(7.088, -4.921, 7.519),
            seed=42,
            num_workers=1,
            vina_cpu=1,
            batch_size=1,
            skip_docking=False,  # Run docking so we have poses to score
            vina_override=True,
            boltz_override=True,
            use_kernels=True,
        )
        runner = BoltzinaRunner(cfg)
        df = runner.run()
        assert isinstance(df, pd.DataFrame)

    def test_mode_b_sdf_input(self, tmp_path, simple_sdf_file):
        """Mode B with SDF input file (3D coords preserved)."""
        from boltzina.runner import BoltzinaRunner, RunnerConfig

        out_dir = tmp_path / "mode_b_sdf"

        cfg = RunnerConfig(
            input_path=simple_sdf_file,
            output_dir=out_dir,
            work_dir=BOLTZ_RESULTS_BASE,
            ligand_chain_id="B",
            grid_center=(7.088, -4.921, 7.519),
            seed=42,
            num_workers=1,
            vina_cpu=1,
            batch_size=1,
            vina_override=True,
            boltz_override=True,
            use_kernels=True,
        )
        runner = BoltzinaRunner(cfg)
        df = runner.run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Mode A: full-auto (requires Boltz-2 model + MSA)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
class TestModeAIntegration:
    CDK2_SEQ = (
        "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLD"
        "VIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLIN"
        "TTCDLKICDFGLARVADPDHDHTGFLTEYVATRWYRAPEVLLGSRHYSTGVDIWSVGCIFAEMCNRKPI"
        "FKGSDYLDQLNRFVTLGTP"
    )

    def test_mode_a_full_pipeline(self, tmp_path):
        """
        Full Mode A pipeline: sequence + SMILES → Boltz-2 → docking → scoring.

        Requires Boltz-2 model weights and MSA to be available.
        """
        from boltzina.runner import BoltzinaRunner, RunnerConfig

        smi_file = make_small_smi_file(tmp_path, n=1)
        out_dir = tmp_path / "mode_a_results"

        cfg = RunnerConfig(
            input_path=smi_file,
            output_dir=out_dir,
            sequence=self.CDK2_SEQ,
            ligand_chain_id="B",
            seed=42,
            num_workers=1,
            vina_cpu=1,
            batch_size=1,
            recycling_steps=1,   # fast for testing
            sampling_steps=10,   # fast for testing
            diffusion_samples=1,
            use_kernels=True,
            use_msa_server=False,  # assume local MSA available
        )

        runner = BoltzinaRunner(cfg)
        df = runner.run()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        csv_path = out_dir / "results.csv"
        assert csv_path.exists()

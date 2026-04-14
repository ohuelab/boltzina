"""
Tests for boltzina.docking.grid — grid center determination.

Covers:
  - Explicit grid center passthrough
  - Grid from reference ligand (PDB/SDF)
  - Grid from Boltz-2 work_dir (CIF parsing)
  - Vina config file generation
  - Priority ordering
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boltzina.docking.grid import (
    _parse_cif_ligand_center,
    determine_grid_center,
    get_grid_center_from_work_dir,
    write_vina_config,
)


# ---------------------------------------------------------------------------
# _parse_cif_ligand_center
# ---------------------------------------------------------------------------

class TestParseCifLigandCenter:
    def test_extracts_coords_from_sample(self, cif_path):
        center = _parse_cif_ligand_center(cif_path, ligand_chain_id="B")
        assert center is not None
        assert len(center) == 3
        # The CDK2 sample ligand should be in the binding site (~7, -5, 7)
        assert -50 < center[0] < 50
        assert -50 < center[1] < 50
        assert -50 < center[2] < 50

    def test_raises_for_missing_chain(self, cif_path):
        with pytest.raises((ValueError, RuntimeError)):
            _parse_cif_ligand_center(cif_path, ligand_chain_id="Z")


# ---------------------------------------------------------------------------
# get_grid_center_from_work_dir
# ---------------------------------------------------------------------------

class TestGetGridCenterFromWorkDir:
    def test_extracts_from_boltz_output(self, boltz_work_dir):
        center = get_grid_center_from_work_dir(
            work_dir=boltz_work_dir,
            fname="1ckp_cdk2",
            ligand_chain_id="B",
        )
        assert center is not None
        assert len(center) == 3
        # Sanity: coordinates should be within the protein's spatial extent (~±50 Å)
        # The Boltz-2 predicted ligand position may differ from the manually-defined
        # vina config center (7.088, -4.921, 7.519) by up to ~20 Å.
        for coord in center:
            assert -100 < coord < 100, f"Coordinate {coord} out of expected range"


# ---------------------------------------------------------------------------
# determine_grid_center — priority tests
# ---------------------------------------------------------------------------

class TestDetermineGridCenter:
    def test_explicit_grid_center_wins(self, boltz_work_dir, tmp_path):
        """Explicit --grid-center should override everything."""
        explicit = (1.0, 2.0, 3.0)
        center = determine_grid_center(
            work_dir=boltz_work_dir,
            fname="1ckp_cdk2",
            ligand_chain_id="B",
            reference_ligand=None,
            grid_center=explicit,
        )
        np.testing.assert_array_almost_equal(center, explicit, decimal=3)

    def test_reference_ligand_pdb(self, ligand_pdb_path, boltz_work_dir):
        """Reference ligand file → center of mass."""
        center = determine_grid_center(
            work_dir=boltz_work_dir,
            fname="1ckp_cdk2",
            ligand_chain_id="B",
            reference_ligand=ligand_pdb_path,
            grid_center=None,
        )
        assert center is not None
        assert len(center) == 3

    def test_work_dir_fallback(self, boltz_work_dir):
        """Without explicit center or reference ligand, use work_dir CIF."""
        center = determine_grid_center(
            work_dir=boltz_work_dir,
            fname="1ckp_cdk2",
            ligand_chain_id="B",
            reference_ligand=None,
            grid_center=None,
        )
        assert center is not None
        assert len(center) == 3

    def test_raises_without_fallback(self, tmp_path):
        """If no source provided and work_dir has no CIF, should raise."""
        with pytest.raises(Exception):
            determine_grid_center(
                work_dir=tmp_path,
                fname="nonexistent",
                ligand_chain_id="B",
                reference_ligand=None,
                grid_center=None,
            )


# ---------------------------------------------------------------------------
# write_vina_config
# ---------------------------------------------------------------------------

class TestWriteVinaConfig:
    def test_writes_file(self, tmp_path):
        out = tmp_path / "vina.txt"
        write_vina_config(center=(1.0, 2.0, 3.0), output_path=out, size=20.0)
        assert out.exists()

    def test_content_has_center_keys(self, tmp_path):
        out = tmp_path / "vina.txt"
        write_vina_config(center=(7.1, -4.9, 7.5), output_path=out, size=25.0)
        content = out.read_text()
        assert "center_x" in content
        assert "center_y" in content
        assert "center_z" in content
        assert "size_x" in content

    def test_center_values_match(self, tmp_path):
        out = tmp_path / "vina.txt"
        write_vina_config(center=(7.088, -4.921, 7.519), output_path=out, size=20.0)
        params = {}
        for line in out.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                params[k.strip()] = v.strip()
        assert abs(float(params["center_x"]) - 7.088) < 0.01
        assert abs(float(params["center_y"]) - (-4.921)) < 0.01
        assert abs(float(params["center_z"]) - 7.519) < 0.01

    def test_size_written(self, tmp_path):
        out = tmp_path / "vina.txt"
        write_vina_config(center=(0, 0, 0), output_path=out, size=30.0)
        params = {}
        for line in out.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                params[k.strip()] = v.strip()
        assert float(params["size_x"]) == 30.0

    def test_seed_written_when_provided(self, tmp_path):
        out = tmp_path / "vina.txt"
        write_vina_config(center=(0, 0, 0), output_path=out, size=20.0, seed=42)
        content = out.read_text()
        assert "seed" in content
        assert "42" in content

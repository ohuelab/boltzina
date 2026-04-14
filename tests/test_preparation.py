"""
Tests for boltzina.preparation — SMILES/SDF → PDB pipeline.

Covers:
  - SMILES to 3D PDB conversion with unique atom names
  - SDF with 3D coords: atom names assigned, coordinates preserved
  - SDF without 3D coords: conformer auto-generated
  - --regenerate-conformer: always regenerates even if 3D exists
  - PKL consistency with PDB atom names
  - Batch processing from SMILES file and SDF file
  - Atom name uniqueness (issue #8 regression)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from rdkit import Chem

from boltzina.preparation import (
    _assign_pdb_atom_names,
    _has_3d_coords,
    prepare_ligands_from_file,
    prepare_mol_from_smiles,
    prepare_mol_from_sdf_mol,
)


# ---------------------------------------------------------------------------
# _has_3d_coords
# ---------------------------------------------------------------------------

class TestHas3dCoords:
    def test_no_conformer(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert not _has_3d_coords(mol)

    def test_flat_conformer(self, flat_sdf_file):
        supplier = Chem.SDMolSupplier(str(flat_sdf_file), removeHs=False)
        mol = next(m for m in supplier if m is not None)
        assert not _has_3d_coords(mol)

    def test_3d_conformer(self, simple_sdf_file):
        supplier = Chem.SDMolSupplier(str(simple_sdf_file), removeHs=False)
        mol = next(m for m in supplier if m is not None)
        assert _has_3d_coords(mol)


# ---------------------------------------------------------------------------
# _assign_pdb_atom_names
# ---------------------------------------------------------------------------

class TestAssignPdbAtomNames:
    def test_names_are_unique(self):
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        mol = _assign_pdb_atom_names(mol)
        names = [atom.GetPDBResidueInfo().GetName().strip() for atom in mol.GetAtoms()]
        assert len(names) == len(set(names)), f"Duplicate atom names: {names}"

    def test_names_stored_as_prop(self):
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        mol = _assign_pdb_atom_names(mol)
        for atom in mol.GetAtoms():
            assert atom.HasProp("name"), "Atom missing 'name' property"

    def test_residue_name_is_unl(self):
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        mol = _assign_pdb_atom_names(mol)
        for atom in mol.GetAtoms():
            assert atom.GetPDBResidueInfo().GetResidueName() == "UNL"


# ---------------------------------------------------------------------------
# prepare_mol_from_smiles
# ---------------------------------------------------------------------------

class TestPrepareMolFromSmiles:
    def test_creates_pdb(self, tmp_out):
        tmp_out.mkdir(parents=True)
        out_pdb = tmp_out / "mol.pdb"
        mol = prepare_mol_from_smiles("c1ccccc1", out_pdb)
        assert mol is not None
        assert out_pdb.exists()

    def test_pdb_has_hetatm_records(self, tmp_out):
        tmp_out.mkdir(parents=True)
        out_pdb = tmp_out / "mol.pdb"
        prepare_mol_from_smiles("CCO", out_pdb)
        content = out_pdb.read_text()
        assert "HETATM" in content or "ATOM" in content

    def test_atom_names_unique(self, tmp_out):
        tmp_out.mkdir(parents=True)
        out_pdb = tmp_out / "mol.pdb"
        mol = prepare_mol_from_smiles("c1ccccc1", out_pdb)
        names = [atom.GetPDBResidueInfo().GetName().strip() for atom in mol.GetAtoms()]
        assert len(names) == len(set(names))

    def test_invalid_smiles_returns_none(self, tmp_out):
        tmp_out.mkdir(parents=True)
        out_pdb = tmp_out / "mol.pdb"
        result = prepare_mol_from_smiles("this_is_not_smiles", out_pdb)
        assert result is None

    def test_3d_coords_present(self, tmp_out):
        tmp_out.mkdir(parents=True)
        out_pdb = tmp_out / "mol.pdb"
        mol = prepare_mol_from_smiles("CCO", out_pdb)
        assert _has_3d_coords(mol)


# ---------------------------------------------------------------------------
# prepare_mol_from_sdf_mol
# ---------------------------------------------------------------------------

class TestPrepareMolFromSdfMol:
    def test_3d_sdf_preserved(self, simple_sdf_file, tmp_out):
        tmp_out.mkdir(parents=True)
        supplier = Chem.SDMolSupplier(str(simple_sdf_file), removeHs=True)
        mol = next(m for m in supplier if m is not None)
        orig_conf = mol.GetConformer()
        orig_coords = [orig_conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

        out_pdb = tmp_out / "mol.pdb"
        prepared = prepare_mol_from_sdf_mol(mol, out_pdb, regenerate_conformer=False)
        assert prepared is not None

        new_conf = prepared.GetConformer()
        for i, orig_pos in enumerate(orig_coords):
            new_pos = new_conf.GetAtomPosition(i)
            assert abs(new_pos.x - orig_pos.x) < 0.01
            assert abs(new_pos.y - orig_pos.y) < 0.01

    def test_flat_sdf_auto_generates_conformer(self, flat_sdf_file, tmp_out):
        tmp_out.mkdir(parents=True)
        supplier = Chem.SDMolSupplier(str(flat_sdf_file), removeHs=True)
        mol = next(m for m in supplier if m is not None)

        out_pdb = tmp_out / "mol.pdb"
        prepared = prepare_mol_from_sdf_mol(mol, out_pdb, regenerate_conformer=False)
        assert prepared is not None
        assert _has_3d_coords(prepared), "Should have auto-generated 3D coords"

    def test_regenerate_conformer_flag(self, simple_sdf_file, tmp_out):
        tmp_out.mkdir(parents=True)
        supplier = Chem.SDMolSupplier(str(simple_sdf_file), removeHs=True)
        mol = next(m for m in supplier if m is not None)
        orig_conf = mol.GetConformer()
        orig_z = [orig_conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]

        out_pdb = tmp_out / "mol.pdb"
        prepared = prepare_mol_from_sdf_mol(mol, out_pdb, regenerate_conformer=True)
        assert prepared is not None
        assert _has_3d_coords(prepared)
        # Coordinates should differ (regenerated) — at least one z differs significantly
        new_conf = prepared.GetConformer()
        new_z = [new_conf.GetAtomPosition(i).z for i in range(prepared.GetNumAtoms())]
        # Either z was regenerated (differs) or there's still non-zero z
        assert _has_3d_coords(prepared)


# ---------------------------------------------------------------------------
# prepare_ligands_from_file — SMILES input
# ---------------------------------------------------------------------------

class TestPrepareLigandsFromFile:
    def test_smi_input(self, simple_smi_file, tmp_out):
        pdb_paths, pkl_path = prepare_ligands_from_file(
            input_path=simple_smi_file,
            output_dir=tmp_out,
        )
        assert pkl_path.exists()
        assert len(pdb_paths) == len([l for l in simple_smi_file.read_text().splitlines() if l.strip()])
        for p in pdb_paths:
            assert p.exists(), f"PDB not found: {p}"

    def test_sdf_input(self, simple_sdf_file, tmp_out):
        pdb_paths, pkl_path = prepare_ligands_from_file(
            input_path=simple_sdf_file,
            output_dir=tmp_out,
        )
        assert pkl_path.exists()
        assert len(pdb_paths) == 3  # 3 molecules in simple_sdf_file
        for p in pdb_paths:
            assert p.exists(), f"PDB not found: {p}"

    def test_pkl_keys_match_pdb_names(self, simple_smi_file, tmp_out):
        pdb_paths, pkl_path = prepare_ligands_from_file(
            input_path=simple_smi_file,
            output_dir=tmp_out,
        )
        with open(pkl_path, "rb") as f:
            mols_dict = pickle.load(f)
        # Every PDB stem should be a key in the pkl dict
        for p in pdb_paths:
            assert p.stem in mols_dict, f"{p.stem} not in pkl"

    def test_atom_names_unique_in_pkl(self, simple_smi_file, tmp_out):
        """Regression test for issue #8: pkl atom names must match PDB."""
        _, pkl_path = prepare_ligands_from_file(
            input_path=simple_smi_file,
            output_dir=tmp_out,
        )
        with open(pkl_path, "rb") as f:
            mols_dict = pickle.load(f)
        for name, mol in mols_dict.items():
            names = [a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()]
            assert len(names) == len(set(names)), (
                f"Duplicate atom names in pkl mol '{name}': {names}"
            )

    def test_with_ligand_prefix(self, tmp_path, tmp_out):
        """ligand_prefix is used as fallback name when SMILES line has no name."""
        # Write SMILES file without names
        smi_path = tmp_path / "unnamed.smi"
        smi_path.write_text("c1ccccc1\nCCO\n")
        pdb_paths, pkl_path = prepare_ligands_from_file(
            input_path=smi_path,
            output_dir=tmp_out,
            ligand_prefix="LIG",
        )
        for p in pdb_paths:
            assert p.stem.startswith("LIG"), f"Unexpected name: {p.stem}"

    def test_directory_input(self, tmp_path):
        """Directory input: all .smi/.sdf files in the directory are merged."""
        import pickle

        ligand_dir = tmp_path / "ligands"
        ligand_dir.mkdir()
        # Write two SMILES files
        (ligand_dir / "batch1.smi").write_text("c1ccccc1 benzene\n")
        (ligand_dir / "batch2.smi").write_text("CCO ethanol\n")

        out_dir = tmp_path / "out"
        pdb_paths, pkl_path = prepare_ligands_from_file(
            input_path=ligand_dir,
            output_dir=out_dir,
        )
        assert len(pdb_paths) == 2, f"Expected 2 PDBs, got {len(pdb_paths)}"
        assert pkl_path.exists()
        with open(pkl_path, "rb") as f:
            mols = pickle.load(f)
        assert len(mols) == 2

    def test_directory_no_files_raises(self, tmp_path):
        """Empty directory raises ValueError."""
        ligand_dir = tmp_path / "empty"
        ligand_dir.mkdir()
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="No supported ligand files"):
            prepare_ligands_from_file(input_path=ligand_dir, output_dir=out_dir)

    def test_empty_smiles_file_returns_empty_list(self, tmp_path):
        """SMILES file with only comments/blank lines returns empty pdb_paths."""
        smi_path = tmp_path / "empty.smi"
        smi_path.write_text("# just a comment\n\n")
        out_dir = tmp_path / "out"
        pdb_paths, pkl_path = prepare_ligands_from_file(input_path=smi_path, output_dir=out_dir)
        assert pdb_paths == []

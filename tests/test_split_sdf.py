"""
Tests for split_batch_sdf_to_pdbs and split_docked_sdf_to_pdbs.

Covers the critical atom-mapping bug where UniDock2 may:
1. Leave explicit hydrogens that removeHs=True did not strip
2. Reorder atoms relative to the input SDF

Both functions must produce PDB files with correct coordinates mapped
back to the template atom names.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from boltzina.docking.unidock2 import split_batch_sdf_to_pdbs, split_docked_sdf_to_pdbs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mol_with_names(smiles: str, seed: int = 42) -> Chem.Mol:
    """
    Build a 3D mol (heavy atoms only) with canonical atom names ("C1", "N2", ...).
    Mimics what preparation.py + pdb_to_sdf produce.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mol, randomSeed=seed)
    assert ok == 0
    mol = Chem.RemoveHs(mol)
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        name = atom.GetSymbol().upper() + str(can_idx + 1)
        atom.SetProp("name", name)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.rjust(4))
        info.SetResidueName("UNL")
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    return mol


def _write_docked_sdf(mol: Chem.Mol, sdf_path: Path, mol_name: str) -> None:
    """Write a single mol to SDF with a given name."""
    w = Chem.SDWriter(str(sdf_path))
    mol.SetProp("_Name", mol_name)
    mol.SetProp("vina_binding_free_energy", "-7.5")
    w.write(mol)
    w.close()


def _get_atom_positions(pdb_path: Path) -> dict[str, tuple[float, float, float]]:
    """Return {atom_name: (x, y, z)} from a PDB file."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
    assert mol is not None, f"Could not read PDB: {pdb_path}"
    result = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info:
            name = info.GetName().strip()
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            result[name] = (pos.x, pos.y, pos.z)
    return result


def _get_template_positions(template_mol: Chem.Mol) -> dict[str, tuple[float, float, float]]:
    """Return {atom_name: (x, y, z)} from a template mol."""
    result = {}
    conf = template_mol.GetConformer()
    for atom in template_mol.GetAtoms():
        name = atom.GetProp("name")
        pos = conf.GetAtomPosition(atom.GetIdx())
        result[name] = (pos.x, pos.y, pos.z)
    return result


# ---------------------------------------------------------------------------
# Tests for split_batch_sdf_to_pdbs
# ---------------------------------------------------------------------------

class TestSplitBatchSdfToPdbs:

    def test_normal_case(self, tmp_path):
        """Same order, no extra Hs — baseline should produce PDB with correct coords."""
        template = _make_mol_with_names("CC(=O)O")  # acetic acid
        docked = copy.deepcopy(template)
        # Give docked a different conformer (simulated docking result)
        AllChem.EmbedMolecule(docked, randomSeed=99)

        sdf_path = tmp_path / "batch.sdf"
        _write_docked_sdf(docked, sdf_path, "MOL_0_unidock2_pose_0")

        out_dir = tmp_path / "out" / "0"
        split_batch_sdf_to_pdbs(sdf_path, [template], [out_dir], num_poses=1)

        pdb_path = out_dir / "docked_ligands" / "docked_ligand_1.pdb"
        assert pdb_path.exists(), "PDB not generated in normal case"

        # Coords must come from docked mol, not template
        docked_positions = _get_template_positions(docked)
        pdb_positions = _get_atom_positions(pdb_path)
        for name, coords in docked_positions.items():
            assert name in pdb_positions, f"Atom {name} missing from PDB"
            assert pytest.approx(pdb_positions[name], abs=1e-3) == coords

    def test_extra_hydrogens_in_docked(self, tmp_path):
        """UniDock2 leaves explicit Hs that removeHs=True did not strip.

        The docked SDF has the heavy atoms plus some explicit Hs attached.
        split_batch_sdf_to_pdbs must still recover the correct heavy-atom coords.
        """
        template = _make_mol_with_names("CC(=O)O")
        # Build docked mol WITH explicit hydrogens
        mol_noH = copy.deepcopy(template)
        AllChem.EmbedMolecule(mol_noH, randomSeed=99)
        # addCoords=True places Hs adjacent to existing heavy-atom positions
        # without moving the heavy atoms themselves
        docked_with_H = Chem.AddHs(mol_noH, addCoords=True)

        sdf_path = tmp_path / "batch_H.sdf"
        _write_docked_sdf(docked_with_H, sdf_path, "MOL_0_unidock2_pose_0")

        out_dir = tmp_path / "out" / "0"
        split_batch_sdf_to_pdbs(sdf_path, [template], [out_dir], num_poses=1)

        pdb_path = out_dir / "docked_ligands" / "docked_ligand_1.pdb"
        assert pdb_path.exists(), "PDB not generated when docked mol has extra Hs"

        # Coords must match the heavy-atom positions from the docked mol
        expected = _get_template_positions(mol_noH)
        actual = _get_atom_positions(pdb_path)
        for name, coords in expected.items():
            assert name in actual, f"Atom {name} missing from PDB"
            assert pytest.approx(actual[name], abs=0.1) == coords

    def test_reordered_atoms(self, tmp_path):
        """UniDock2 reorders heavy atoms relative to the input SDF.

        RenumberAtoms simulates a permuted output. Coords must still land on
        the correct (by atom name) template atoms.
        """
        template = _make_mol_with_names("CC(=O)O")
        n = template.GetNumAtoms()
        AllChem.EmbedMolecule(template, randomSeed=42)

        # Simulate docking: same molecule, different conformer, shuffled atom order
        docked_base = copy.deepcopy(template)
        AllChem.EmbedMolecule(docked_base, randomSeed=99)

        # Shuffle atoms (reverse order as a simple permutation)
        perm = list(reversed(range(n)))
        docked_shuffled = Chem.RenumberAtoms(docked_base, perm)

        sdf_path = tmp_path / "batch_reorder.sdf"
        _write_docked_sdf(docked_shuffled, sdf_path, "MOL_0_unidock2_pose_0")

        out_dir = tmp_path / "out" / "0"
        split_batch_sdf_to_pdbs(sdf_path, [template], [out_dir], num_poses=1)

        pdb_path = out_dir / "docked_ligands" / "docked_ligand_1.pdb"
        assert pdb_path.exists(), "PDB not generated when docked mol has reordered atoms"

        # Coords must match docked_base (pre-shuffle) positions by atom name
        expected = _get_template_positions(docked_base)
        actual = _get_atom_positions(pdb_path)
        for name, coords in expected.items():
            assert name in actual, f"Atom {name} missing from PDB"
            assert pytest.approx(actual[name], abs=1e-3) == coords

    def test_completely_different_molecule_skipped(self, tmp_path):
        """A docked mol that shares no substructure with template must be skipped."""
        template = _make_mol_with_names("c1ccccc1")  # benzene (6C aromatic)
        other = _make_mol_with_names("CCN")           # ethylamine (completely different)
        AllChem.EmbedMolecule(other, randomSeed=42)

        sdf_path = tmp_path / "batch_mismatch.sdf"
        _write_docked_sdf(other, sdf_path, "MOL_0_unidock2_pose_0")

        out_dir = tmp_path / "out" / "0"
        split_batch_sdf_to_pdbs(sdf_path, [template], [out_dir], num_poses=1)

        pdb_path = out_dir / "docked_ligands" / "docked_ligand_1.pdb"
        assert not pdb_path.exists(), "PDB must NOT be generated for mismatched molecule"

        # Scores file should exist but be empty for this ligand
        scores_path = out_dir / "unidock2_scores.json"
        assert scores_path.exists()
        scores = json.loads(scores_path.read_text())
        assert scores == {}, f"Scores should be empty dict, got: {scores}"

    def test_batch_multiple_ligands(self, tmp_path):
        """Multiple ligands in a single batch — each must get its own PDB."""
        smiles_list = ["CC(=O)O", "c1ccccc1", "CCO"]
        templates = [_make_mol_with_names(smi, seed=i) for i, smi in enumerate(smiles_list)]
        for t in templates:
            AllChem.EmbedMolecule(t, randomSeed=42)

        # Write one docked pose per ligand
        w = Chem.SDWriter(str(tmp_path / "batch_multi.sdf"))
        for i, t in enumerate(templates):
            d = copy.deepcopy(t)
            AllChem.EmbedMolecule(d, randomSeed=99 + i)
            d.SetProp("_Name", f"MOL_{i}_unidock2_pose_0")
            d.SetProp("vina_binding_free_energy", f"{-5.0 - i}")
            w.write(d)
        w.close()

        out_dirs = [tmp_path / "out" / str(i) for i in range(len(templates))]
        split_batch_sdf_to_pdbs(
            tmp_path / "batch_multi.sdf", templates, out_dirs, num_poses=1
        )

        for i, out_dir in enumerate(out_dirs):
            pdb_path = out_dir / "docked_ligands" / "docked_ligand_1.pdb"
            assert pdb_path.exists(), f"PDB missing for ligand {i}"

    def test_scores_written(self, tmp_path):
        """Docking score from SDF property must be written to unidock2_scores.json."""
        template = _make_mol_with_names("CCO")
        docked = copy.deepcopy(template)
        AllChem.EmbedMolecule(docked, randomSeed=99)

        sdf_path = tmp_path / "batch_score.sdf"
        _write_docked_sdf(docked, sdf_path, "MOL_0_unidock2_pose_0")

        out_dir = tmp_path / "out" / "0"
        split_batch_sdf_to_pdbs(sdf_path, [template], [out_dir], num_poses=1)

        scores_path = out_dir / "unidock2_scores.json"
        assert scores_path.exists()
        scores = json.loads(scores_path.read_text())
        assert "1" in scores
        assert pytest.approx(scores["1"]) == -7.5


# ---------------------------------------------------------------------------
# Tests for split_docked_sdf_to_pdbs (single-ligand path)
# ---------------------------------------------------------------------------

class TestSplitDockedSdfToPdbs:

    def test_normal_case(self, tmp_path):
        """Same order, no extra Hs."""
        template = _make_mol_with_names("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        docked = copy.deepcopy(template)
        AllChem.EmbedMolecule(docked, randomSeed=77)

        sdf_path = tmp_path / "docked.sdf"
        _write_docked_sdf(docked, sdf_path, "pose_0")

        results = split_docked_sdf_to_pdbs(sdf_path, template, tmp_path, num_poses=1)

        assert len(results) == 1
        pdb_path, score = results[0]
        assert Path(pdb_path).exists()
        assert pytest.approx(score) == -7.5

    def test_extra_hydrogens_in_docked(self, tmp_path):
        """Explicit Hs in docked mol must not prevent coordinate transfer."""
        template = _make_mol_with_names("CC(=O)Oc1ccccc1C(=O)O")
        mol_noH = copy.deepcopy(template)
        AllChem.EmbedMolecule(mol_noH, randomSeed=77)
        docked_with_H = Chem.AddHs(mol_noH, addCoords=True)

        sdf_path = tmp_path / "docked_H.sdf"
        _write_docked_sdf(docked_with_H, sdf_path, "pose_0")

        results = split_docked_sdf_to_pdbs(sdf_path, template, tmp_path, num_poses=1)

        assert len(results) == 1
        pdb_path, _ = results[0]
        assert Path(pdb_path).exists()

    def test_reordered_atoms(self, tmp_path):
        """Shuffled atom order in docked mol must produce correct coords by atom name."""
        template = _make_mol_with_names("CC(=O)Oc1ccccc1C(=O)O")
        n = template.GetNumAtoms()
        AllChem.EmbedMolecule(template, randomSeed=42)

        docked_base = copy.deepcopy(template)
        AllChem.EmbedMolecule(docked_base, randomSeed=77)
        perm = list(reversed(range(n)))
        docked_shuffled = Chem.RenumberAtoms(docked_base, perm)

        sdf_path = tmp_path / "docked_reorder.sdf"
        _write_docked_sdf(docked_shuffled, sdf_path, "pose_0")

        results = split_docked_sdf_to_pdbs(sdf_path, template, tmp_path, num_poses=1)

        assert len(results) == 1
        pdb_path, _ = results[0]
        assert Path(pdb_path).exists()

        expected = _get_template_positions(docked_base)
        actual = _get_atom_positions(pdb_path)
        for name, coords in expected.items():
            assert name in actual
            assert pytest.approx(actual[name], abs=1e-3) == coords

    def test_mismatch_skipped(self, tmp_path):
        """Completely different molecule must be skipped (no PDB, empty results)."""
        template = _make_mol_with_names("c1ccccc1")
        other = _make_mol_with_names("CCN")
        AllChem.EmbedMolecule(other, randomSeed=42)

        sdf_path = tmp_path / "docked_mismatch.sdf"
        _write_docked_sdf(other, sdf_path, "pose_0")

        results = split_docked_sdf_to_pdbs(sdf_path, template, tmp_path, num_poses=1)

        assert results == [], f"Expected empty results for mismatched mol, got {results}"

    def test_num_poses_limit(self, tmp_path):
        """num_poses=1 must return only the first pose from a multi-pose SDF."""
        template = _make_mol_with_names("CCO")
        AllChem.EmbedMolecule(template, randomSeed=42)

        sdf_path = tmp_path / "multi_pose.sdf"
        w = Chem.SDWriter(str(sdf_path))
        for seed in [10, 20, 30]:
            d = copy.deepcopy(template)
            AllChem.EmbedMolecule(d, randomSeed=seed)
            d.SetProp("vina_binding_free_energy", "-6.0")
            w.write(d)
        w.close()

        results = split_docked_sdf_to_pdbs(sdf_path, template, tmp_path, num_poses=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Regression tests using real MF-PCBA PDBs that failed with bond-order mismatch
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_template_from_pdb(pdb_path: Path) -> Chem.Mol:
    """Load template mol exactly as engine.py does: try sanitize=True, fall back to False."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
    assert mol is not None, f"Could not load {pdb_path}"
    return mol


def _make_clean_docked(template_mol: Chem.Mol, seed: int = 42) -> Chem.Mol | None:
    """Simulate a clean docked mol via InChi roundtrip (mimics UniDock2 re-sanitization)."""
    from rdkit.Chem.inchi import MolFromInchi, MolToInchi
    try:
        inchi = MolToInchi(template_mol)
        clean = MolFromInchi(inchi)
        if clean is None:
            return None
        clean_H = Chem.AddHs(clean)
        ok = AllChem.EmbedMolecule(clean_H, randomSeed=seed)
        return clean_H if ok == 0 else None
    except Exception:
        return None


def _write_single_docked_sdf(mol: Chem.Mol, path: Path, lig_idx: int = 0) -> None:
    w = Chem.SDWriter(str(path))
    mol.SetProp("_Name", f"MOL_{lig_idx}_unidock2_pose_0")
    mol.SetProp("vina_binding_free_energy", "-7.0")
    w.write(mol)
    w.close()


class TestRealPDBRegressions:
    """
    Regression tests using actual MF-PCBA ligands that triggered bond-order mismatch
    failures in production (1053173-743445 fold 2, ~9300 failures).

    2826617.pdb — template=15 atoms, docked=27 atoms (broken CONECT: sanitize=False fallback)
    2826618.pdb — template=17 atoms, docked=17 atoms (same count, different bond orders)
    2827088.pdb — template=25 atoms: UniDock2 strips phosphonate ring → 19 atoms (unrescuable)
    """

    def test_bond_order_mismatch_15_27_recoverable(self, tmp_path):
        """2826617: extra Hs + bond-order mismatch → should recover via AdjustQueryProperties."""
        pdb_path = FIXTURES_DIR / "2826617.pdb"
        if not pdb_path.exists():
            pytest.skip("fixture not available")
        template = _load_template_from_pdb(pdb_path)
        docked = _make_clean_docked(template)
        if docked is None:
            pytest.skip("InChi roundtrip unavailable for this mol")

        sdf_path = tmp_path / "docked.sdf"
        _write_single_docked_sdf(docked, sdf_path, lig_idx=0)
        out_dirs = [tmp_path / "out0"]
        out_dirs[0].mkdir()

        split_batch_sdf_to_pdbs(sdf_path, [template], out_dirs, num_poses=1)

        pdb_files = list((out_dirs[0] / "docked_ligands").glob("docked_ligand_*.pdb"))
        assert len(pdb_files) == 1, "Bond-order mismatch case should be recovered"

    def test_bond_order_mismatch_17_17_recoverable(self, tmp_path):
        """2826618: same atom count but different bond orders → should recover."""
        pdb_path = FIXTURES_DIR / "2826618.pdb"
        if not pdb_path.exists():
            pytest.skip("fixture not available")
        template = _load_template_from_pdb(pdb_path)
        docked = _make_clean_docked(template)
        if docked is None:
            pytest.skip("InChi roundtrip unavailable for this mol")

        sdf_path = tmp_path / "docked.sdf"
        _write_single_docked_sdf(docked, sdf_path, lig_idx=0)
        out_dirs = [tmp_path / "out0"]
        out_dirs[0].mkdir()

        split_batch_sdf_to_pdbs(sdf_path, [template], out_dirs, num_poses=1)

        pdb_files = list((out_dirs[0] / "docked_ligands").glob("docked_ligand_*.pdb"))
        assert len(pdb_files) == 1, "Bond-order mismatch case should be recovered"

    def test_atom_loss_unrescuable(self, tmp_path):
        """2827088: UniDock2 strips ring atoms (25→19) → should skip gracefully."""
        pdb_path = FIXTURES_DIR / "2827088.pdb"
        if not pdb_path.exists():
            pytest.skip("fixture not available")
        template = _load_template_from_pdb(pdb_path)
        # Simulate atom-loss: build a smaller mol (fewer heavy atoms than template)
        smaller = Chem.MolFromSmiles("c1ccccc1")  # 6 atoms vs 25 in template
        assert smaller is not None
        smaller_H = Chem.AddHs(smaller)
        AllChem.EmbedMolecule(smaller_H, randomSeed=42)

        sdf_path = tmp_path / "docked.sdf"
        _write_single_docked_sdf(smaller_H, sdf_path, lig_idx=0)
        out_dirs = [tmp_path / "out0"]
        out_dirs[0].mkdir()

        split_batch_sdf_to_pdbs(sdf_path, [template], out_dirs, num_poses=1)

        pdb_files = list((out_dirs[0] / "docked_ligands").glob("docked_ligand_*.pdb"))
        assert len(pdb_files) == 0, "Atom-loss case should be skipped (not rescued)"

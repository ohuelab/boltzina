"""
Shared test fixtures for Boltzina test suite.

All tests are expected to run in a GPU environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_CDK2 = REPO_ROOT / "sample" / "CDK2"
BOLTZ_RESULTS_BASE = SAMPLE_CDK2 / "boltz_results_base"


@pytest.fixture(scope="session")
def sample_cdk2_dir() -> Path:
    """Path to sample/CDK2/ directory."""
    assert SAMPLE_CDK2.exists(), f"Sample CDK2 directory not found: {SAMPLE_CDK2}"
    return SAMPLE_CDK2


@pytest.fixture(scope="session")
def boltz_work_dir() -> Path:
    """Path to the pre-computed Boltz-2 output for CDK2."""
    assert BOLTZ_RESULTS_BASE.exists(), f"Boltz results not found: {BOLTZ_RESULTS_BASE}"
    return BOLTZ_RESULTS_BASE


@pytest.fixture(scope="session")
def receptor_pdb(boltz_work_dir) -> Path:
    p = boltz_work_dir / "predictions" / "1ckp_cdk2" / "1ckp_cdk2_model_0_protein.pdb"
    assert p.exists(), f"Receptor PDB not found: {p}"
    return p


@pytest.fixture(scope="session")
def cif_path(boltz_work_dir) -> Path:
    p = boltz_work_dir / "predictions" / "1ckp_cdk2" / "1ckp_cdk2_model_0.cif"
    assert p.exists(), f"CIF file not found: {p}"
    return p


@pytest.fixture(scope="session")
def ligand_pdb_path(boltz_work_dir) -> Path:
    p = boltz_work_dir / "predictions" / "1ckp_cdk2" / "1ckp_cdk2_model_0_ligand.pdb"
    assert p.exists(), f"Ligand PDB not found: {p}"
    return p


# ---------------------------------------------------------------------------
# Simple molecular test data
# ---------------------------------------------------------------------------

# CDK2 inhibitor SMILES (roscovitine)
ROSCOVITINE_SMILES = "CCN(CC)c1nc(Nc2ccccc2)c2ncn(C(C)CO)c2n1"

# Simple molecules for unit tests
SIMPLE_SMILES_LIST = [
    ("c1ccccc1", "benzene"),
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic_acid"),
]


@pytest.fixture
def tmp_out(tmp_path) -> Path:
    """Temporary output directory for test artifacts."""
    return tmp_path / "out"


@pytest.fixture
def simple_smi_file(tmp_path) -> Path:
    """Write a small SMILES file and return its path."""
    smi_path = tmp_path / "test_ligands.smi"
    lines = [f"{smi} {name}" for smi, name in SIMPLE_SMILES_LIST]
    smi_path.write_text("\n".join(lines) + "\n")
    return smi_path


@pytest.fixture
def simple_sdf_file(tmp_path) -> Path:
    """Write a small SDF file (with 3D conformers) and return its path."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    sdf_path = tmp_path / "test_ligands.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for smiles, name in SIMPLE_SMILES_LIST:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    return sdf_path


@pytest.fixture
def flat_sdf_file(tmp_path) -> Path:
    """Write an SDF file WITHOUT 3D coordinates (all z=0) and return its path."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import rdGeometry

    sdf_path = tmp_path / "flat_ligands.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for smiles, name in SIMPLE_SMILES_LIST[:1]:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
        # Flatten: zero all z-coordinates
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x, pos.y, 0.0))
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    return sdf_path

"""
Ligand preparation pipeline for Boltzina.

Converts SMILES strings or SDF files into PDB files with unique atom names,
and produces a prepared_mols.pkl dict for consistent atom ordering during
Boltz-2 affinity scoring.

Supported inputs:
  - SMILES file (.smi / .txt): one "SMILES name" per line
  - SDF file (.sdf): multi-molecule SD file
  - Single SMILES string (programmatic use)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import compute_3d_conformer

# Preserve all atom properties when pickling (required for name/PDBResidueInfo)
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def _parse_smiles_file_line(
    line: str,
    line_no: int,
    ligand_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parse one SMILES-file record into a single-token SMILES and ligand name.

    RDKit can parse records of the form "SMILES [CXSMILES] [name]" when
    allowCXSMILES and parseName are enabled. Using it here prevents CXSMILES
    enhanced-stereo annotations such as "|&1:1,11|" from being mistaken for
    the ligand name.
    """
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = True
    params.parseName = True
    mol = Chem.MolFromSmiles(line, params)

    if mol is None:
        parts = line.split()
        smiles = parts[0]
        if len(parts) < 2:
            if ligand_prefix is None:
                raise ValueError(
                    f"Ligand name missing on line {line_no} and --ligand-prefix not set."
                )
            name = f"{ligand_prefix}{line_no - 1}"
        else:
            name = parts[1]
        return smiles, name

    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    if not name:
        if ligand_prefix is None:
            raise ValueError(
                f"Ligand name missing on line {line_no} and --ligand-prefix not set."
            )
        name = f"{ligand_prefix}{line_no - 1}"
    else:
        name = name.split()[0]

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles, name


def _has_3d_coords(mol: Mol) -> bool:
    """Return True if the molecule has non-trivial 3D coordinates."""
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    # Check if any z-coordinate is non-zero
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        if abs(pos.z) > 1e-6:
            return True
    return False


def _assign_pdb_atom_names(mol: Mol) -> Mol:
    """
    Assign unique PDB-compatible atom names to all heavy atoms.

    Uses canonical atom ordering (element + canonical_index) to ensure
    consistent naming between PDB files and the prepared_mols.pkl dict.
    This prevents the atom-order mismatch bug reported in issue #8.
    """
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    pdb_resn, pdb_chain, pdb_resi = "UNL", "A", 1
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"Atom name too long (>4 chars): {atom_name}. "
                f"This may happen with very large molecules."
            )
        atom.SetProp("name", atom_name)
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
        info.SetName(atom_name.rjust(4))
        info.SetResidueName(pdb_resn)
        info.SetResidueNumber(pdb_resi)
        info.SetChainId(pdb_chain)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    return mol


def prepare_mol_from_smiles(smiles: str, output_path: str | Path) -> Optional[Mol]:
    """
    Convert a SMILES string to a 3D PDB file with unique atom names.

    Returns the prepared RDKit Mol (with atom names set) for inclusion
    in prepared_mols.pkl, or None on failure.
    """
    output_path = Path(output_path)
    try:
        mol_2d = Chem.MolFromSmiles(smiles)
        if mol_2d is None:
            print(f"Failed to parse SMILES: {smiles}")
            return None
        mol_add_h = Chem.AddHs(mol_2d)
        success = compute_3d_conformer(mol_add_h)
        if not success:
            print(f"Failed to compute 3D conformer for SMILES: {smiles}")
        mol_3d = Chem.RemoveHs(mol_add_h)
        mol_3d = _assign_pdb_atom_names(mol_3d)
        Chem.MolToPDBFile(mol_3d, str(output_path))
        return mol_3d
    except Exception as e:
        print(f"Failed to prepare mol {output_path}: {e}")
        return None


def prepare_mol_from_sdf_mol(
    sdf_mol: Mol,
    output_path: str | Path,
    regenerate_conformer: bool = False,
) -> Optional[Mol]:
    """
    Convert an RDKit Mol read from SDF into a PDB file with unique atom names.

    3D coordinate handling:
      - If the molecule has 3D coordinates (non-zero z) and regenerate_conformer
        is False: use the existing coordinates as-is.
      - Otherwise: run compute_3d_conformer to generate new 3D coordinates.

    Returns the prepared Mol for inclusion in prepared_mols.pkl, or None on failure.
    """
    output_path = Path(output_path)
    try:
        mol = Chem.RemoveHs(sdf_mol)
        if mol is None:
            print(f"RDKit failed to process SDF molecule for {output_path}")
            return None

        has_3d = _has_3d_coords(mol)
        if not has_3d or regenerate_conformer:
            mol_with_h = Chem.AddHs(mol)
            # Remove existing conformers so EmbedMolecule starts fresh.
            # Without this, a flat (z=0) conformer from SDF blocks re-embedding.
            mol_with_h.RemoveAllConformers()
            success = compute_3d_conformer(mol_with_h)
            if not success:
                print(f"Failed to compute 3D conformer for {output_path}")
            mol = Chem.RemoveHs(mol_with_h)

        mol = _assign_pdb_atom_names(mol)
        Chem.MolToPDBFile(mol, str(output_path))
        return mol
    except Exception as e:
        print(f"Failed to prepare SDF mol {output_path}: {e}")
        return None


def prepare_ligands_from_smiles_file(
    smi_path: str | Path,
    output_dir: str | Path,
    ligand_prefix: Optional[str] = None,
) -> Tuple[List[Path], Path]:
    """
    Prepare all ligands from a SMILES file.

    File format: one "SMILES [CXSMILES] name" per line
    (name is optional if ligand_prefix is set).

    Returns:
        (pdb_paths, pkl_path): list of prepared PDB paths and path to prepared_mols.pkl
    """
    smi_path = Path(smi_path)
    output_dir = Path(output_dir)
    (output_dir / "input_pdbs").mkdir(parents=True, exist_ok=True)

    smiles_dict: Dict[str, str] = {}
    with open(smi_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles, name = _parse_smiles_file_line(line, i + 1, ligand_prefix)
            smiles_dict[name] = smiles

    return _prepare_smiles_dict(smiles_dict, output_dir)


def prepare_ligands_from_smiles_list(
    smiles_list: List[str],
    names: List[str],
    output_dir: str | Path,
) -> Tuple[List[Path], Path]:
    """
    Prepare ligands from a list of SMILES strings and names.

    Returns:
        (pdb_paths, pkl_path)
    """
    output_dir = Path(output_dir)
    (output_dir / "input_pdbs").mkdir(parents=True, exist_ok=True)
    smiles_dict = dict(zip(names, smiles_list))
    return _prepare_smiles_dict(smiles_dict, output_dir)


def _prepare_smiles_dict(
    smiles_dict: Dict[str, str],
    output_dir: Path,
) -> Tuple[List[Path], Path]:
    mols_dict: Dict[str, Mol] = {}
    pdb_paths: List[Path] = []

    for name, smiles in smiles_dict.items():
        output_path = output_dir / "input_pdbs" / f"{name}.pdb"
        mol = prepare_mol_from_smiles(smiles, output_path)
        if mol is None:
            continue
        mols_dict[name] = mol
        pdb_paths.append(output_path)

    pkl_path = output_dir / "prepared_mols.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(mols_dict, f)

    print(f"Prepared {len(mols_dict)}/{len(smiles_dict)} molecules")
    return pdb_paths, pkl_path


def prepare_ligands_from_sdf(
    sdf_path: str | Path,
    output_dir: str | Path,
    ligand_prefix: str = "lig",
    regenerate_conformer: bool = False,
) -> Tuple[List[Path], Path]:
    """
    Prepare all ligands from an SDF file.

    Molecule names are taken from the SDF _Name field; if empty, uses
    ligand_prefix + index.

    Returns:
        (pdb_paths, pkl_path)
    """
    sdf_path = Path(sdf_path)
    output_dir = Path(output_dir)
    (output_dir / "input_pdbs").mkdir(parents=True, exist_ok=True)

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols_dict: Dict[str, Mol] = {}
    pdb_paths: List[Path] = []

    for i, mol in enumerate(supplier):
        if mol is None:
            print(f"Warning: molecule {i} in {sdf_path} could not be read, skipping.")
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        if not name:
            name = f"{ligand_prefix}{i}"

        # Deduplicate names
        if name in mols_dict:
            name = f"{name}_{i}"

        output_path = output_dir / "input_pdbs" / f"{name}.pdb"
        prepared = prepare_mol_from_sdf_mol(
            mol, output_path, regenerate_conformer=regenerate_conformer
        )
        if prepared is None:
            continue
        mols_dict[name] = prepared
        pdb_paths.append(output_path)

    pkl_path = output_dir / "prepared_mols.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(mols_dict, f)

    print(f"Prepared {len(mols_dict)} molecules from SDF")
    return pdb_paths, pkl_path


def prepare_ligands_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    ligand_prefix: Optional[str] = None,
    regenerate_conformer: bool = False,
) -> Tuple[List[Path], Path]:
    """
    Auto-detect input format and prepare ligands.

    Supported formats:
      - .smi / .txt : SMILES list
      - .sdf        : SD file
      - directory   : all .smi, .txt, .sdf files found (merged)

    Returns:
        (pdb_paths, pkl_path)
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        return _prepare_ligands_from_directory(
            input_path, output_dir, ligand_prefix, regenerate_conformer
        )

    suffix = input_path.suffix.lower()
    if suffix in (".smi", ".txt"):
        return prepare_ligands_from_smiles_file(input_path, output_dir, ligand_prefix)
    elif suffix == ".sdf":
        prefix = ligand_prefix or input_path.stem
        return prepare_ligands_from_sdf(
            input_path, output_dir, ligand_prefix=prefix,
            regenerate_conformer=regenerate_conformer,
        )
    else:
        raise ValueError(
            f"Unsupported ligand file format: {suffix}. "
            f"Supported: .smi, .txt, .sdf"
        )


def _prepare_ligands_from_directory(
    input_dir: Path,
    output_dir: str | Path,
    ligand_prefix: Optional[str],
    regenerate_conformer: bool,
) -> Tuple[List[Path], Path]:
    """
    Prepare all ligand files found in a directory.

    Searches for .smi, .txt, and .sdf files and processes each one,
    merging results into a single prepared_mols.pkl.

    Raises ValueError if no supported files are found.
    """
    output_dir = Path(output_dir)
    (output_dir / "input_pdbs").mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for pattern in ("*.smi", "*.txt", "*.sdf"):
        files.extend(sorted(input_dir.glob(pattern)))

    if not files:
        raise ValueError(
            f"No supported ligand files (.smi, .txt, .sdf) found in directory: {input_dir}"
        )

    all_pdb_paths: List[Path] = []
    merged_mols: Dict[str, Mol] = {}

    for f in files:
        suffix = f.suffix.lower()
        if suffix in (".smi", ".txt"):
            pdb_paths, pkl = prepare_ligands_from_smiles_file(f, output_dir, ligand_prefix)
        else:
            prefix = ligand_prefix or f.stem
            pdb_paths, pkl = prepare_ligands_from_sdf(
                f, output_dir, ligand_prefix=prefix,
                regenerate_conformer=regenerate_conformer,
            )
        all_pdb_paths.extend(pdb_paths)
        with open(pkl, "rb") as fh:
            merged_mols.update(pickle.load(fh))

    pkl_path = output_dir / "prepared_mols.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(merged_mols, fh)

    return all_pdb_paths, pkl_path

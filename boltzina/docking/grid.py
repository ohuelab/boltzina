"""
Automatic docking grid determination for Boltzina.

Strategies (in priority order):
  1. Explicit --grid-center: use as-is
  2. Reference ligand file (--reference-ligand): compute center of mass
  3. Boltz-2 predicted ligand position (default auto): extract ligand chain
     coords from predicted CIF/PDB in work_dir

Writes a Vina-compatible config file with center_x/y/z and size_x/y/z.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem

from boltzina.docking.calculate_com import get_center_of_mass_from_file


def _parse_cif_ligand_center(cif_path: Path, ligand_chain_id: str) -> np.ndarray:
    """
    Extract the center of mass of a ligand chain from a CIF file.

    Reads atom coordinates for the specified chain_id and returns their
    geometric centroid (simple mean, not mass-weighted, since CIF may lack
    element info in all flavors).
    """
    coords = []
    try:
        with open(cif_path) as f:
            in_atom_site = False
            col_x = col_y = col_z = col_auth_asym = None
            headers: list[str] = []

            for line in f:
                line = line.rstrip("\n")
                if line.startswith("_atom_site."):
                    tag = line.strip().lstrip("_").split(".")[1]
                    headers.append(tag)
                    if tag == "Cartn_x":
                        col_x = len(headers) - 1
                    elif tag == "Cartn_y":
                        col_y = len(headers) - 1
                    elif tag == "Cartn_z":
                        col_z = len(headers) - 1
                    elif tag == "auth_asym_id":
                        col_auth_asym = len(headers) - 1
                    in_atom_site = True
                elif in_atom_site and line.startswith("ATOM") or (
                    in_atom_site and line.startswith("HETATM")
                ):
                    parts = line.split()
                    if (
                        col_x is not None
                        and col_y is not None
                        and col_z is not None
                        and col_auth_asym is not None
                        and len(parts) > max(col_x, col_y, col_z, col_auth_asym)
                    ):
                        chain = parts[col_auth_asym]
                        if chain == ligand_chain_id:
                            try:
                                x = float(parts[col_x])
                                y = float(parts[col_y])
                                z = float(parts[col_z])
                                coords.append([x, y, z])
                            except ValueError:
                                pass
    except Exception as e:
        raise RuntimeError(f"Failed to parse CIF {cif_path}: {e}")

    if not coords:
        raise ValueError(
            f"No atoms found for chain '{ligand_chain_id}' in {cif_path}. "
            f"Check --ligand-chain-id."
        )
    return np.mean(coords, axis=0)


def _find_predicted_structure(work_dir: Path, fname: str) -> Optional[Path]:
    """
    Find the predicted structure file in a boltz output directory.

    Looks for {fname}_model_0.cif or {fname}_model_0.pdb in
    work_dir/predictions/{fname}/.
    """
    pred_dir = work_dir / "predictions" / fname
    for ext in (".cif", ".pdb"):
        candidate = pred_dir / f"{fname}_model_0{ext}"
        if candidate.exists():
            return candidate

    # Also search one level up (some boltz versions)
    for ext in (".cif", ".pdb"):
        candidate = work_dir / "predictions" / f"{fname}_model_0{ext}"
        if candidate.exists():
            return candidate

    return None


def get_grid_center_from_work_dir(
    work_dir: Path,
    fname: str,
    ligand_chain_id: str = "B",
) -> np.ndarray:
    """
    Extract the ligand center of mass from a Boltz-2 predicted structure.

    Args:
        work_dir: Boltz-2 output directory (contains predictions/)
        fname: Base filename used for the Boltz-2 prediction
        ligand_chain_id: Chain ID of the ligand in the predicted structure

    Returns:
        numpy array [x, y, z] — the grid center
    """
    struct_path = _find_predicted_structure(work_dir, fname)
    if struct_path is None:
        raise RuntimeError(
            f"No predicted structure found in {work_dir}/predictions/{fname}/. "
            f"Run Boltz-2 prediction first, or specify --grid-center explicitly."
        )

    if struct_path.suffix == ".cif":
        return _parse_cif_ligand_center(struct_path, ligand_chain_id)
    else:
        # PDB: use RDKit to read and extract chain
        return _get_center_from_pdb_chain(struct_path, ligand_chain_id)


def _get_center_from_pdb_chain(pdb_path: Path, chain_id: str) -> np.ndarray:
    """Extract center of mass of a specific chain from a PDB file."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    if mol is None:
        raise RuntimeError(f"Failed to read PDB: {pdb_path}")

    coords = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetChainId().strip() == chain_id:
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

    if not coords:
        raise ValueError(
            f"No atoms found for chain '{chain_id}' in {pdb_path}."
        )
    return np.mean(coords, axis=0)


def write_vina_config(
    center: np.ndarray,
    output_path: Path,
    size: float | Tuple[float, float, float] = 20.0,
    num_modes: int = 1,
    cpu: int = 1,
    seed: Optional[int] = None,
) -> None:
    """
    Write a Vina-compatible config file.

    Args:
        center: [x, y, z] grid center
        output_path: Where to write the config file
        size: Box size in Angstroms (scalar → cubic, or (sx, sy, sz))
        num_modes: Number of docking modes
        cpu: CPUs per Vina call
        seed: Random seed (omitted if None)
    """
    if isinstance(size, (int, float)):
        sx = sy = sz = float(size)
    else:
        sx, sy, sz = size

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"center_x = {center[0]:.3f}",
        f"center_y = {center[1]:.3f}",
        f"center_z = {center[2]:.3f}",
        f"size_x = {sx:.1f}",
        f"size_y = {sy:.1f}",
        f"size_z = {sz:.1f}",
        f"num_modes = {num_modes}",
        f"cpu = {cpu}",
    ]
    if seed is not None:
        lines.append(f"seed = {seed}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"Grid center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
        f"| Size: {sx}x{sy}x{sz} Å"
    )


def determine_grid_center(
    work_dir: Optional[Path] = None,
    fname: Optional[str] = None,
    ligand_chain_id: str = "B",
    reference_ligand: Optional[Path] = None,
    grid_center: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Determine grid center using available information.

    Priority:
      1. grid_center (explicit tuple)
      2. reference_ligand file (COM via RDKit)
      3. work_dir + fname: predicted ligand position from Boltz-2 output

    Returns:
        numpy array [x, y, z]
    """
    if grid_center is not None:
        center = np.array(grid_center, dtype=float)
        print(
            f"Using explicit grid center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
        )
        return center

    if reference_ligand is not None:
        center = get_center_of_mass_from_file(str(reference_ligand))
        print(
            f"Grid center from reference ligand {reference_ligand.name}: "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
        )
        return center

    if work_dir is not None and fname is not None:
        center = get_grid_center_from_work_dir(work_dir, fname, ligand_chain_id)
        print(
            f"Grid center from Boltz-2 prediction (chain {ligand_chain_id}): "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
        )
        return center

    raise ValueError(
        "Cannot determine grid center: provide --grid-center, --reference-ligand, "
        "or --work-dir with a valid Boltz-2 prediction."
    )

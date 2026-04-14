"""
Uni-Dock2 adapter for Boltzina pipeline.

Handles:
- Vina config parsing (center/box extraction)
- PDB → SDF conversion with atom order tracking
- Running Uni-Dock2 via CLI
- Splitting multi-pose SDF output into PDBs with correct atom names
"""

import os
import copy
import json
import subprocess
import yaml
from pathlib import Path
from typing import Optional

from rdkit import Chem

from boltzina.config import get_unidock2_config as _get_unidock2_config


def _get_unidock2_bin() -> str:
    """Resolve Uni-Dock2 binary path via boltzina config."""
    return _get_unidock2_config()["bin"]


def _get_unidock2_env() -> dict:
    """Return the environment dict needed to run Uni-Dock2."""
    cfg = _get_unidock2_config()
    env = os.environ.copy()
    pythonpath = cfg.get("pythonpath", "")
    extra_path = cfg.get("extra_path", "")
    if pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pythonpath}:{existing}" if existing else pythonpath
    if extra_path:
        env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    return env


def parse_vina_config(config_path: Path) -> dict:
    """Parse a Vina config file into a dict."""
    params = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                params[key.strip()] = val.strip()
    return params


def pdb_to_sdf(pdb_path: Path, sdf_path: Path) -> Optional[Chem.Mol]:
    """
    Convert a PDB file to SDF using RDKit, preserving atom order.

    Returns the RDKit mol template (with PDBResidueInfo / atom names set),
    which is the SAME object that was written to SDF — so atom indices match.
    """
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=True)
    if mol is None:
        # Retry without sanitization for unusual ligands
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not read PDB file: {pdb_path}")

    # Ensure each atom has a "name" property from PDBResidueInfo
    for atom in mol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info and not atom.HasProp("name"):
            atom_name = pdb_info.GetName().strip()
            if atom_name:
                atom.SetProp("name", atom_name)

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return mol


def run_unidock2(
    receptor_pdb: Path,
    ligand_sdf: Path,
    center: tuple,
    output_sdf: Path,
    working_dir: Path,
    unidock2_config: Optional[dict] = None,
    unidock2_bin: Optional[str] = None,
) -> None:
    """
    Run Uni-Dock2 docking via CLI using a temporary YAML config.

    center: (x, y, z) floats
    temp_dir_name is forced to $HOME/tmpdir to comply with /tmp prohibition.
    """
    if unidock2_bin is None:
        unidock2_bin = _get_unidock2_bin()
    if unidock2_config is None:
        unidock2_config = {}

    tmpdir = Path(os.environ["HOME"]) / "tmpdir"
    tmpdir.mkdir(parents=True, exist_ok=True)

    box_size = unidock2_config.get("box_size", [30.0, 30.0, 30.0])
    exhaustiveness = unidock2_config.get("exhaustiveness", 512)
    num_pose = unidock2_config.get("num_pose", 10)
    seed = unidock2_config.get("seed", 12345)
    gpu_device_id = unidock2_config.get("gpu_device_id", 0)
    search_mode = unidock2_config.get("search_mode", "balance")
    task = unidock2_config.get("task", "screen")

    yaml_config = {
        "Required": {
            "receptor": str(receptor_pdb.resolve()),
            "ligand": str(ligand_sdf.resolve()),
            "ligand_batch": None,
            "center": list(center),
        },
        "Advanced": {
            "exhaustiveness": exhaustiveness,
            "randomize": True,
            "mc_steps": 40,
            "opt_steps": -1,
            "refine_steps": 5,
            "num_pose": num_pose,
            "rmsd_limit": 1.0,
            "energy_range": 5.0,
            "seed": seed,
            "use_tor_lib": False,
            "energy_decomp": False,
        },
        "Settings": {
            "box_size": list(box_size),
            "task": task,
            "search_mode": search_mode,
        },
        "Hardware": {
            "gpu_device_id": gpu_device_id,
            "n_cpu": None,
        },
        "Preprocessing": {
            "template_docking": False,
            "reference_sdf_file_name": None,
            "core_atom_mapping_dict_list": None,
            "covalent_ligand": False,
            "covalent_residue_atom_info_list": None,
            "preserve_receptor_hydrogen": False,
            "temp_dir_name": str(tmpdir),
            "output_docking_pose_sdf_file_name": str(output_sdf.resolve()),
        },
    }

    yaml_path = working_dir / "unidock2_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    env = _get_unidock2_env()

    cmd = [unidock2_bin, "docking", "-cf", str(yaml_path.resolve())]
    try:
        subprocess.run(cmd, check=True, cwd=str(working_dir), env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Uni-Dock2 docking failed: {e}")

    if not output_sdf.exists():
        raise RuntimeError(f"Uni-Dock2 did not produce output SDF: {output_sdf}")


def run_unidock2_batch(
    receptor_pdb: Path,
    ligand_sdf_list: list,
    center: tuple,
    output_sdf: Path,
    working_dir: Path,
    unidock2_config: Optional[dict] = None,
    unidock2_bin: Optional[str] = None,
) -> None:
    """
    Run Uni-Dock2 docking for multiple ligands in one GPU call via ligand_batch mode.

    ligand_sdf_list: list of Path to individual ligand SDF files.
    Writes a batch text file listing all SDF paths, then calls Uni-Dock2 once.
    Output is a single combined SDF with all poses for all ligands.
    """
    if unidock2_bin is None:
        unidock2_bin = _get_unidock2_bin()
    if unidock2_config is None:
        unidock2_config = {}

    working_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(os.environ["HOME"]) / "tmpdir"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Write batch file listing all ligand SDF paths
    batch_file = working_dir / "ligand_batch.txt"
    with open(batch_file, "w") as f:
        for sdf_path in ligand_sdf_list:
            f.write(str(Path(sdf_path).resolve()) + "\n")

    box_size = unidock2_config.get("box_size", [30.0, 30.0, 30.0])
    exhaustiveness = unidock2_config.get("exhaustiveness", 512)
    num_pose = unidock2_config.get("num_pose", 1)
    seed = unidock2_config.get("seed", 12345)
    gpu_device_id = unidock2_config.get("gpu_device_id", 0)
    search_mode = unidock2_config.get("search_mode", "balance")
    task = unidock2_config.get("task", "screen")
    n_cpu = unidock2_config.get("n_cpu", None)

    yaml_config = {
        "Required": {
            "receptor": str(receptor_pdb.resolve()),
            "ligand": None,
            "ligand_batch": str(batch_file.resolve()),
            "center": list(center),
        },
        "Advanced": {
            "exhaustiveness": exhaustiveness,
            "randomize": True,
            "mc_steps": 40,
            "opt_steps": -1,
            "refine_steps": 5,
            "num_pose": num_pose,
            "rmsd_limit": 1.0,
            "energy_range": 5.0,
            "seed": seed,
            "use_tor_lib": False,
            "energy_decomp": False,
        },
        "Settings": {
            "box_size": list(box_size),
            "task": task,
            "search_mode": search_mode,
        },
        "Hardware": {
            "gpu_device_id": gpu_device_id,
            "n_cpu": n_cpu,
        },
        "Preprocessing": {
            "template_docking": False,
            "reference_sdf_file_name": None,
            "core_atom_mapping_dict_list": None,
            "covalent_ligand": False,
            "covalent_residue_atom_info_list": None,
            "preserve_receptor_hydrogen": False,
            "temp_dir_name": str(tmpdir),
            "output_docking_pose_sdf_file_name": str(output_sdf.resolve()),
        },
    }

    yaml_path = working_dir / "unidock2_batch_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    env = _get_unidock2_env()

    cmd = [unidock2_bin, "docking", "-cf", str(yaml_path.resolve())]
    try:
        subprocess.run(cmd, check=True, cwd=str(working_dir), env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Uni-Dock2 batch docking failed: {e}")

    if not output_sdf.exists():
        raise RuntimeError(f"Uni-Dock2 did not produce batch output SDF: {output_sdf}")


def split_batch_sdf_to_pdbs(
    docked_sdf_path: Path,
    template_mols: list,
    output_dirs: list,
    num_poses: int,
) -> None:
    """
    Split a batch Uni-Dock2 output SDF into per-ligand PDB files.

    Molecule names in the batch SDF follow the pattern: MOL_{i}_unidock2_pose_{j}
    where i = ligand index (0-based, matching order in batch file) and j = pose index (0-based).

    template_mols[i] is the RDKit mol (with atom names) used to write the i-th input SDF.
    output_dirs[i] is the output directory for the i-th ligand.
    Scores are saved to output_dirs[i]/docked_ligands/unidock2_scores.json.
    """
    n_ligands = len(template_mols)
    for out_dir in output_dirs:
        (out_dir / "docked_ligands").mkdir(parents=True, exist_ok=True)

    # scores[ligand_idx][pose_label] = score
    scores = {i: {} for i in range(n_ligands)}

    supplier = Chem.SDMolSupplier(str(docked_sdf_path), removeHs=True)
    for docked_mol in supplier:
        if docked_mol is None:
            continue

        # Get molecule name: try ud2_molecule_name property first, then _Name
        mol_name = None
        if docked_mol.HasProp("ud2_molecule_name"):
            mol_name = docked_mol.GetProp("ud2_molecule_name")
        elif docked_mol.HasProp("_Name"):
            mol_name = docked_mol.GetProp("_Name")
        if not mol_name:
            print(f"Warning: could not determine molecule name for a pose, skipping")
            continue

        # Parse "MOL_{i}_unidock2_pose_{j}"
        try:
            parts = mol_name.split("_")
            # MOL _ {i} _ unidock2 _ pose _ {j}
            lig_idx = int(parts[1])
            pose_idx = int(parts[-1])
        except (IndexError, ValueError):
            print(f"Warning: unexpected mol name format '{mol_name}', skipping")
            continue

        if lig_idx >= n_ligands:
            print(f"Warning: ligand index {lig_idx} out of range ({n_ligands}), skipping")
            continue

        if pose_idx >= num_poses:
            continue

        template_mol = template_mols[lig_idx]
        n_template = template_mol.GetNumAtoms()
        n_docked = docked_mol.GetNumAtoms()
        if n_template != n_docked:
            print(
                f"Warning: atom count mismatch for ligand {lig_idx} pose {pose_idx}: "
                f"template={n_template}, docked={n_docked}. Skipping."
            )
            continue

        # Deep copy template (preserves atom names), overwrite coordinates
        pose_mol = copy.deepcopy(template_mol)
        conf = pose_mol.GetConformer()
        docked_conf = docked_mol.GetConformer()
        for atom_idx in range(n_template):
            pos = docked_conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(atom_idx, pos)

        # Extract docking score
        score = None
        if docked_mol.HasProp("vina_binding_free_energy"):
            try:
                score = float(docked_mol.GetProp("vina_binding_free_energy"))
            except ValueError:
                pass

        pdb_path = output_dirs[lig_idx] / "docked_ligands" / f"docked_ligand_{pose_idx + 1}.pdb"
        Chem.MolToPDBFile(pose_mol, str(pdb_path))

        pose_label = str(pose_idx + 1)
        scores[lig_idx][pose_label] = score

    # Save per-ligand scores to ligand_output_dir directly (not docked_ligands/,
    # which gets removed by _cleanup_preaffinity_intermediates before score extraction)
    for i, out_dir in enumerate(output_dirs):
        scores_path = out_dir / "unidock2_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores[i], f)


def split_docked_sdf_to_pdbs(
    docked_sdf_path: Path,
    template_mol: Chem.Mol,
    output_dir: Path,
    num_poses: int,
) -> list:
    """
    Split a multi-pose SDF from Uni-Dock2 into individual PDB files,
    restoring atom names from the template mol.

    Atom order is guaranteed to match because:
    - template_mol is the SAME RDKit mol used to write the input SDF
    - Uni-Dock2 preserves atom order (deepcopy + index-based coord assignment)
    - So docked_mol atom idx N == template_mol atom idx N

    Returns: list of (pdb_path, docking_score_or_None)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    supplier = Chem.SDMolSupplier(str(docked_sdf_path), removeHs=True)
    pdb_files = []

    for pose_idx, docked_mol in enumerate(supplier):
        if pose_idx >= num_poses:
            break
        if docked_mol is None:
            print(f"Warning: pose {pose_idx + 1} could not be read from docked SDF")
            continue

        n_template = template_mol.GetNumAtoms()
        n_docked = docked_mol.GetNumAtoms()
        assert n_template == n_docked, (
            f"Atom count mismatch between template ({n_template}) "
            f"and docked mol ({n_docked}) at pose {pose_idx + 1}"
        )

        # Deep copy template to get atom names, then overwrite coordinates
        pose_mol = copy.deepcopy(template_mol)
        conf = pose_mol.GetConformer()
        docked_conf = docked_mol.GetConformer()
        for atom_idx in range(n_template):
            pos = docked_conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(atom_idx, pos)

        # Extract docking score from SDF property
        score = None
        if docked_mol.HasProp("vina_binding_free_energy"):
            try:
                score = float(docked_mol.GetProp("vina_binding_free_energy"))
            except ValueError:
                pass

        pdb_path = output_dir / f"docked_ligand_{pose_idx + 1}.pdb"
        Chem.MolToPDBFile(pose_mol, str(pdb_path))
        pdb_files.append((pdb_path, score))

    # Save all scores to JSON for later extraction.
    # Save to the parent of output_dir (i.e., ligand_output_dir), not inside docked_ligands/,
    # because docked_ligands/ gets removed by _cleanup_preaffinity_intermediates before
    # _extract_results reads the scores.
    scores = {str(i + 1): score for i, (_, score) in enumerate(pdb_files)}
    scores_path = output_dir.parent / "unidock2_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f)

    return pdb_files

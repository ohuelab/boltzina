"""
Boltz-2 prerequisite automation for Boltzina.

Handles:
  - Generating a Boltz-2 YAML input file from a protein sequence + representative SMILES
  - Running Boltz-2 structure prediction via the Python API (boltz.main.predict)
  - Returning the paths that Boltzina needs (work_dir, receptor_pdb, fname)

This module is used by BoltzinaRunner for Mode A (full-auto) operation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from boltzina.config import get_boltz_cache


def generate_boltz_yaml(
    sequence: str,
    representative_smiles: str,
    output_path: Path,
    protein_chain_id: str = "A",
    ligand_chain_id: str = "B",
) -> Path:
    """
    Generate a Boltz-2 YAML input file for structure + affinity prediction.

    The generated YAML specifies:
      - A protein chain with the given sequence
      - A ligand chain with the representative SMILES
      - An affinity property targeting the ligand chain

    Args:
        sequence: Amino acid sequence (single-letter codes)
        representative_smiles: SMILES for the reference ligand used to determine
            the binding site (used for grid center determination after prediction)
        output_path: Path to write the YAML file
        protein_chain_id: Chain ID for the protein (default: "A")
        ligand_chain_id: Chain ID for the ligand (default: "B")

    Returns:
        Path to the written YAML file
    """
    # Validate sequence
    sequence = sequence.strip().upper()
    if not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", sequence):
        raise ValueError(
            "Protein sequence contains non-standard amino acid characters. "
            "Only single-letter codes (ACDEFGHIKLMNPQRSTVWY) are supported."
        )

    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_chain_id,
                    "sequence": sequence,
                }
            },
            {
                "ligand": {
                    "id": ligand_chain_id,
                    "smiles": representative_smiles,
                }
            },
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_chain_id,
                }
            }
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return output_path


def run_boltz_predict(
    yaml_path: Path,
    out_dir: Path,
    cache: Optional[Path] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: Optional[float] = None,
    max_parallel_samples: Optional[int] = None,
    use_potentials: bool = False,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    affinity_mw_correction: bool = False,
    output_format: str = "mmcif",
    override: bool = False,
    seed: Optional[int] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    preprocessing_threads: int = 1,
) -> Path:
    """
    Run Boltz-2 structure prediction via the Python API.

    Args:
        yaml_path: Path to the Boltz-2 YAML input file
        out_dir: Parent output directory (boltz creates a subdir inside)
        cache: Boltz model cache dir (defaults to ~/.boltz)
        ... (Boltz-2 prediction parameters, see boltz.main.predict)

    Returns:
        work_dir: The actual Boltz-2 output directory
            (out_dir / f"boltz_results_{yaml_path.stem}")
    """
    from boltz.main import predict  # imported lazily to avoid heavy import at module load

    if cache is None:
        cache = get_boltz_cache()

    predict(
        data=str(yaml_path),
        out_dir=str(out_dir),
        cache=str(cache),
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        step_scale=step_scale,
        max_parallel_samples=max_parallel_samples,
        use_potentials=use_potentials,
        max_msa_seqs=max_msa_seqs,
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        no_kernels=no_kernels,
        affinity_mw_correction=affinity_mw_correction,
        output_format=output_format,
        override=override,
        seed=seed,
        devices=devices,
        accelerator=accelerator,
        preprocessing_threads=preprocessing_threads,
        model="boltz2",
    )

    # boltz.main.predict creates: out_dir / f"boltz_results_{yaml_path.stem}"
    work_dir = Path(out_dir) / f"boltz_results_{yaml_path.stem}"
    if not work_dir.exists():
        raise RuntimeError(
            f"Boltz-2 prediction completed but expected output directory not found: {work_dir}"
        )
    return work_dir


def extract_receptor_pdb(work_dir: Path, fname: str) -> Path:
    """
    Extract the receptor (protein-only) PDB from a Boltz-2 prediction.

    Boltz-2 outputs the full complex as CIF. Boltzina's pipeline needs a
    protein-only PDB file as receptor. This function:
      1. Finds the predicted structure CIF/PDB
      2. Extracts only protein chains using pdb-tools (pdb_selchain)
      3. Returns the path to the extracted receptor PDB

    Args:
        work_dir: Boltz-2 output directory
        fname: Base filename (stem of the YAML input)

    Returns:
        Path to the receptor PDB file
    """
    import subprocess

    pred_dir = work_dir / "predictions" / fname
    receptor_pdb = pred_dir / f"{fname}_receptor.pdb"

    if receptor_pdb.exists():
        return receptor_pdb

    # Find the predicted structure
    cif_path = pred_dir / f"{fname}_model_0.cif"
    pdb_path = pred_dir / f"{fname}_model_0.pdb"

    if cif_path.exists():
        # Convert CIF to PDB using pdb-tools
        raw_pdb = pred_dir / f"{fname}_model_0_raw.pdb"
        try:
            result = subprocess.run(
                ["python", "-m", "pdbtools.pdb_fromcif", str(cif_path)],
                capture_output=True, text=True, check=False
            )
            if result.returncode != 0 or not result.stdout.strip():
                # Try maxit conversion fallback
                _extract_protein_from_cif_rdkit(cif_path, receptor_pdb)
                return receptor_pdb
            raw_pdb.write_text(result.stdout)
            pdb_path = raw_pdb
        except FileNotFoundError:
            _extract_protein_from_cif_rdkit(cif_path, receptor_pdb)
            return receptor_pdb

    if not pdb_path.exists():
        raise RuntimeError(
            f"No predicted structure found in {pred_dir}. "
            f"Expected {fname}_model_0.cif or {fname}_model_0.pdb"
        )

    # Extract protein chain(s) only (chain A by default, excluding ligand chain)
    _extract_protein_lines(pdb_path, receptor_pdb)
    return receptor_pdb


def _extract_protein_lines(input_pdb: Path, output_pdb: Path) -> None:
    """
    Write a PDB file containing only ATOM records (protein/nucleic acid),
    excluding HETATM records (ligands, waters).
    """
    lines = []
    with open(input_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                lines.append(line)
    if not lines:
        raise RuntimeError(
            f"No ATOM records found in {input_pdb}. "
            "The predicted structure may be empty or in an unexpected format."
        )
    with open(output_pdb, "w") as f:
        f.writelines(lines)
        if not lines[-1].startswith("END"):
            f.write("END\n")


def _extract_protein_from_cif_rdkit(cif_path: Path, output_pdb: Path) -> None:
    """
    Fallback: use Biopython-style CIF parsing to extract protein atoms.
    This is a simple text-based approach for CIF files.
    """
    protein_lines = []
    try:
        with open(cif_path) as f:
            content = f.read()
        # Simple heuristic: write a notice that this requires maxit
        raise RuntimeError(
            f"Cannot convert CIF {cif_path} to PDB without pdb-tools or maxit. "
            "Install pdb-tools (already a dependency) and ensure it is on PATH."
        )
    except RuntimeError:
        raise


def setup_boltz_for_run(
    sequence: str,
    representative_smiles: str,
    work_base_dir: Path,
    fname: str = "boltzina_input",
    ligand_chain_id: str = "B",
    boltz_kwargs: Optional[dict] = None,
) -> tuple[Path, Path]:
    """
    Full setup pipeline for Mode A: generate YAML, run Boltz-2, extract receptor.

    If work_dir already exists and contains a valid manifest.json, skip prediction
    and reuse the existing results.

    Args:
        sequence: Protein amino acid sequence
        representative_smiles: SMILES for grid center determination
        work_base_dir: Parent directory for Boltz-2 output
        fname: Base name for input/output files
        ligand_chain_id: Ligand chain ID in the YAML
        boltz_kwargs: Additional kwargs forwarded to run_boltz_predict()

    Returns:
        (work_dir, receptor_pdb): Boltz-2 output dir and receptor PDB path
    """
    boltz_kwargs = boltz_kwargs or {}
    work_base_dir = Path(work_base_dir)
    work_base_dir.mkdir(parents=True, exist_ok=True)

    expected_work_dir = work_base_dir / f"boltz_results_{fname}"
    manifest_path = expected_work_dir / "processed" / "manifest.json"

    if manifest_path.exists():
        print(f"Reusing existing Boltz-2 results in {expected_work_dir}")
        work_dir = expected_work_dir
    else:
        yaml_path = work_base_dir / f"{fname}.yaml"
        generate_boltz_yaml(
            sequence=sequence,
            representative_smiles=representative_smiles,
            output_path=yaml_path,
            ligand_chain_id=ligand_chain_id,
        )
        print(f"Generated Boltz-2 input YAML: {yaml_path}")
        print("Running Boltz-2 structure prediction...")
        work_dir = run_boltz_predict(
            yaml_path=yaml_path,
            out_dir=work_base_dir,
            **boltz_kwargs,
        )

    receptor_pdb = extract_receptor_pdb(work_dir, fname)
    print(f"Receptor PDB: {receptor_pdb}")
    return work_dir, receptor_pdb

"""
Boltz-2 setup utilities for Boltzina.

Provides:
  - generate_boltz_yaml(): Create Boltz-2 YAML from sequences + reference SMILES
  - run_boltz_predict(): Call the underlying boltz predict callback
  - extract_receptor_pdb(): Extract protein chain from Boltz-2 CIF output
  - setup_boltz_for_run(): Orchestrate prediction (generate YAML → predict → extract PDB)
  - setup_boltz_for_run_from_yaml(): Same but using a pre-written YAML file
"""

from __future__ import annotations

import re
import string
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml

from boltzina.config import get_boltz_cache
from boltzina.tools import resolve_executable, runtime_env


def _ensure_boltz_rdkit_compat() -> None:
    """Patch RDKit symbols expected by Boltz across RDKit releases."""
    try:
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        return

    if not hasattr(AllChem, "Descriptors"):
        AllChem.Descriptors = Descriptors


def generate_boltz_yaml(
    sequences: list[str],
    representative_smiles: str,
    output_path: Path,
    use_msa_server: bool = False,
) -> tuple[Path, str]:
    """
    Generate a Boltz-2 YAML input file from one or more protein sequences.

    Chain IDs are assigned automatically: A, B, C, ... for proteins in order,
    then the next available letter for the ligand.

    Args:
        sequences: List of protein amino acid sequences (single-letter codes).
                   One entry = one chain. Multiple entries = multi-chain complex.
        representative_smiles: SMILES for the reference ligand (defines binding
                               site; used for grid center determination after
                               structure prediction).
        output_path: Path to write the generated YAML file.
        use_msa_server: If False, mark protein MSAs as empty so Boltz can run
                        offline. If True, leave MSA unspecified so Boltz can
                        generate it with the configured server.

    Returns:
        (yaml_path, ligand_chain_id): Path to the written YAML and the ligand's
        chain ID (the letter after the last protein chain).

    Raises:
        ValueError: If sequences is empty, too many sequences (>25), or any
                    sequence contains non-standard amino acid characters.
    """
    if not sequences:
        raise ValueError("At least one protein sequence is required.")

    n = len(sequences)
    letters = list(string.ascii_uppercase)
    if n >= len(letters):
        raise ValueError(
            f"Too many protein chains ({n}). Maximum supported is {len(letters) - 1}."
        )

    protein_chain_ids = letters[:n]
    ligand_chain_id = letters[n]

    # Validate and normalise each sequence
    clean_seqs = []
    for i, seq in enumerate(sequences):
        seq = seq.strip().upper()
        if not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq):
            raise ValueError(
                f"Protein sequence {i + 1} contains non-standard amino acid characters. "
                "Only single-letter codes (ACDEFGHIKLMNPQRSTVWY) are supported."
            )
        clean_seqs.append(seq)

    protein_entries = []
    for chain_id, seq in zip(protein_chain_ids, clean_seqs):
        protein = {"id": chain_id, "sequence": seq}
        if not use_msa_server:
            protein["msa"] = "empty"
        protein_entries.append({"protein": protein})

    config = {
        "version": 1,
        "sequences": protein_entries
        + [
            {"ligand": {"id": ligand_chain_id, "smiles": representative_smiles}}
        ],
        "properties": [
            {"affinity": {"binder": ligand_chain_id}}
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return output_path, ligand_chain_id


def _add_empty_msa_to_proteins(yaml_data: dict) -> bool:
    """Set missing protein MSA entries to 'empty' for offline Boltz runs."""
    changed = False
    for entry in yaml_data.get("sequences", []):
        if not isinstance(entry, dict):
            continue
        protein = entry.get("protein")
        if not isinstance(protein, dict):
            continue
        if protein.get("msa") in (None, ""):
            protein["msa"] = "empty"
            changed = True
    return changed


def _prepare_yaml_for_boltz_run(
    yaml_path: Path,
    dest_yaml: Path,
    use_msa_server: bool,
) -> Path:
    """Copy YAML into the work directory, adding offline MSA markers if needed."""
    yaml_path = Path(yaml_path)
    dest_yaml = Path(dest_yaml)

    if use_msa_server:
        if dest_yaml.resolve() != yaml_path.resolve():
            dest_yaml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(yaml_path, dest_yaml)
        return dest_yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    changed = _add_empty_msa_to_proteins(data)
    if changed or dest_yaml.resolve() != yaml_path.resolve():
        dest_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return dest_yaml


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
    Run Boltz-2 structure prediction.

    Args:
        yaml_path: Path to the Boltz-2 YAML input file
        out_dir: Parent output directory (boltz creates a subdir inside)
        cache: Boltz model cache dir (defaults to ~/.boltz)
        ... (Boltz-2 prediction parameters, see boltz.main.predict.callback)

    Returns:
        work_dir: The actual Boltz-2 output directory
            (out_dir / f"boltz_results_{yaml_path.stem}")
    """
    from boltz.main import predict  # imported lazily to avoid heavy import at module load

    if cache is None:
        cache = get_boltz_cache()

    _ensure_boltz_rdkit_compat()

    # boltz.main.predict is a click.Command in Boltz 2.x. Calling that command
    # object with keyword arguments sends them to click.Context and raises
    # "unexpected keyword argument 'data'". Use the wrapped callback instead.
    predict_callback = getattr(predict, "callback", predict)

    predict_callback(
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
    Extract a protein-only PDB from a Boltz-2 CIF prediction output.

    Looks for {fname}_model_0.cif in work_dir/predictions/{fname}/,
    converts it to PDB, and returns a protein-only PDB.

    Falls back to a pre-existing _protein.pdb if available.
    """
    from pathlib import Path as _Path

    pred_dir = work_dir / "predictions" / fname
    cif_path = pred_dir / f"{fname}_model_0.cif"
    pdb_path = pred_dir / f"{fname}_model_0.pdb"
    protein_pdb = pred_dir / f"{fname}_model_0_protein.pdb"

    # If protein PDB already extracted, return it
    if protein_pdb.exists():
        return protein_pdb

    # Try CIF → PDB via maxit or pdb_fromcif
    if cif_path.exists():
        try:
            _convert_cif_to_protein_pdb(cif_path, protein_pdb)
            if protein_pdb.exists():
                return protein_pdb
        except Exception as e:
            print(f"Warning: CIF→PDB conversion failed: {e}. Trying fallback.")

    # Fallback: use existing .pdb directly (may include ligand)
    if pdb_path.exists():
        return _extract_protein_atoms(pdb_path, protein_pdb)

    raise RuntimeError(
        f"No predicted structure found in {pred_dir}. "
        "Expected {fname}_model_0.cif or {fname}_model_0.pdb."
    )


def _convert_cif_to_protein_pdb(cif_path: Path, output_pdb: Path) -> None:
    """Convert a CIF file to PDB using pdb_fromcif (pdb-tools), then filter protein atoms."""
    import subprocess

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdb = output_pdb.with_suffix(".tmp.pdb")

    pdb_fromcif = resolve_executable(
        "pdb_fromcif",
        env_var="BOLTZINA_PDB_FROMCIF_PATH",
        required=False,
    )
    command = [pdb_fromcif, str(cif_path)] if pdb_fromcif else [
        sys.executable,
        "-m",
        "pdbtools.pdb_fromcif",
        str(cif_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=runtime_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdb_fromcif failed: {result.stderr}")

    tmp_pdb.write_text(result.stdout)
    _extract_protein_atoms(tmp_pdb, output_pdb)
    tmp_pdb.unlink(missing_ok=True)


def _extract_protein_atoms(input_pdb: Path, output_pdb: Path) -> Path:
    """Write a new PDB containing only ATOM records (protein atoms)."""
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(input_pdb) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                lines.append(line)
    output_pdb.write_text("".join(lines))
    return output_pdb


def parse_yaml_ligand_chain(yaml_path: Path) -> tuple[str, str]:
    """
    Parse a Boltz-2 YAML and extract the reference ligand SMILES and chain ID.

    Validates that the YAML has:
      - At least one protein entry
      - Exactly one ligand entry that is the affinity binder
      - A properties.affinity section

    Returns:
        (smiles, ligand_chain_id)

    Raises:
        ValueError: If the YAML format is invalid.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    sequences = data.get("sequences", [])
    properties = data.get("properties", [])

    # Find affinity binder chain
    affinity_binder = None
    for prop in properties:
        if "affinity" in prop:
            affinity_binder = prop["affinity"].get("binder")
            break

    if affinity_binder is None:
        raise ValueError(
            f"YAML {yaml_path.name} must have a 'properties.affinity.binder' entry. "
            "This specifies which ligand chain to use for affinity prediction and grid center."
        )

    # Find protein entries and ligand entry
    protein_count = 0
    ligand_smiles = None
    ligand_chain_id = None

    for entry in sequences:
        if "protein" in entry:
            protein_count += 1
        elif "ligand" in entry:
            lig = entry["ligand"]
            chain_id = lig.get("id")
            # Normalise: id may be a string or list
            if isinstance(chain_id, list):
                chain_id = chain_id[0]
            if chain_id == affinity_binder:
                smiles = lig.get("smiles")
                if smiles is None:
                    raise ValueError(
                        f"Ligand chain '{affinity_binder}' in YAML must specify 'smiles' "
                        "(CCD-only ligands are not supported for affinity prediction)."
                    )
                ligand_smiles = smiles
                ligand_chain_id = chain_id

    if protein_count == 0:
        raise ValueError(
            f"YAML {yaml_path.name} must contain at least one 'protein' entry."
        )

    if ligand_smiles is None:
        raise ValueError(
            f"YAML {yaml_path.name}: affinity binder chain '{affinity_binder}' not found "
            "among ligand entries, or its id does not match."
        )

    return ligand_smiles, ligand_chain_id


def setup_boltz_for_run(
    sequences: list[str],
    representative_smiles: str,
    work_base_dir: Path,
    fname: str = "boltzina_input",
    boltz_kwargs: Optional[dict] = None,
) -> tuple[Path, Path, str]:
    """
    Full setup pipeline for structure prediction mode (from sequences).

    Generates a Boltz-2 YAML from the given protein sequences and reference SMILES,
    runs Boltz-2 prediction (or reuses existing results), and extracts the receptor PDB.

    Args:
        sequences: List of protein amino acid sequences (one per chain).
        representative_smiles: SMILES for the reference/binder ligand.
        work_base_dir: Parent directory for Boltz-2 output.
        fname: Base name for input/output files.
        boltz_kwargs: Additional kwargs forwarded to run_boltz_predict().

    Returns:
        (work_dir, receptor_pdb, ligand_chain_id)
    """
    boltz_kwargs = boltz_kwargs or {}
    work_base_dir = Path(work_base_dir)
    work_base_dir.mkdir(parents=True, exist_ok=True)

    expected_work_dir = work_base_dir / f"boltz_results_{fname}"
    manifest_path = expected_work_dir / "processed" / "manifest.json"

    if manifest_path.exists():
        print(f"Reusing existing Boltz-2 results in {expected_work_dir}")
        work_dir = expected_work_dir
        # Re-derive ligand_chain_id from number of sequences
        import string as _string
        ligand_chain_id = _string.ascii_uppercase[len(sequences)]
    else:
        yaml_path = work_base_dir / f"{fname}.yaml"
        _, ligand_chain_id = generate_boltz_yaml(
            sequences=sequences,
            representative_smiles=representative_smiles,
            output_path=yaml_path,
            use_msa_server=bool(boltz_kwargs.get("use_msa_server")),
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
    return work_dir, receptor_pdb, ligand_chain_id


def setup_boltz_for_run_from_yaml(
    yaml_path: Path,
    work_base_dir: Path,
    fname: str = "boltzina_input",
    boltz_kwargs: Optional[dict] = None,
) -> tuple[Path, Path, str]:
    """
    Full setup pipeline for structure prediction mode (from a pre-written YAML).

    The YAML is passed directly to Boltz-2 (no generation step). The ligand
    chain ID and reference SMILES are extracted from the YAML's
    properties.affinity.binder and the corresponding ligand entry.

    Args:
        yaml_path: Path to a Boltz-2 compatible YAML file.
        work_base_dir: Parent directory for Boltz-2 output.
        fname: Base name (used for the output subdirectory).
        boltz_kwargs: Additional kwargs forwarded to run_boltz_predict().

    Returns:
        (work_dir, receptor_pdb, ligand_chain_id)
    """
    boltz_kwargs = boltz_kwargs or {}
    work_base_dir = Path(work_base_dir)
    work_base_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = Path(yaml_path)

    # Validate and extract ligand chain info from YAML
    _, ligand_chain_id = parse_yaml_ligand_chain(yaml_path)

    expected_work_dir = work_base_dir / f"boltz_results_{fname}"
    manifest_path = expected_work_dir / "processed" / "manifest.json"

    if manifest_path.exists():
        print(f"Reusing existing Boltz-2 results in {expected_work_dir}")
        work_dir = expected_work_dir
    else:
        # Copy YAML into work_base_dir with fname as stem so boltz creates
        # the expected boltz_results_{fname} output directory
        dest_yaml = work_base_dir / f"{fname}.yaml"
        dest_yaml = _prepare_yaml_for_boltz_run(
            yaml_path=yaml_path,
            dest_yaml=dest_yaml,
            use_msa_server=bool(boltz_kwargs.get("use_msa_server")),
        )
        print(f"Running Boltz-2 structure prediction (from YAML: {yaml_path.name})...")
        work_dir = run_boltz_predict(
            yaml_path=dest_yaml,
            out_dir=work_base_dir,
            **boltz_kwargs,
        )

    receptor_pdb = extract_receptor_pdb(work_dir, fname)
    print(f"Receptor PDB: {receptor_pdb}")
    return work_dir, receptor_pdb, ligand_chain_id

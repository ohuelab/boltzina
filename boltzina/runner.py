"""
BoltzinaRunner — end-to-end orchestrator for `boltzina run`.

Two pipelines:

  Structure prediction pipeline (--sequence / --sequence-file / --yaml):
    1. Resolve protein sequences (single/multi-chain) or use pre-written YAML
    2. Determine reference SMILES for Boltz-2 complex prediction
    3. Run Boltz-2 structure prediction (or reuse existing)
    4. Extract receptor PDB from Boltz-2 output
    5. Determine grid center from predicted ligand position
    6. Prepare ligands from SMILES/SDF input
    7. Run docking + scoring
    8. Save results CSV

  Rescore pipeline (--work-dir):
    1. Extract receptor PDB from existing Boltz-2 output (or use --receptor-pdb)
    2. Determine grid center from predicted ligand position (or --grid-center)
    3. Prepare ligands from SMILES/SDF input
    4. Run docking + scoring
    5. Save results CSV
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from boltzina.docking.grid import determine_grid_center, write_vina_config
from boltzina.preparation import prepare_ligands_from_file


@dataclass
class RunnerConfig:
    """Configuration for BoltzinaRunner."""

    # --- Input ---
    input_path: Path  # Ligand file (.smi, .sdf) or directory
    output_dir: Path

    # --- Protein input: provide exactly one of these ---
    sequence: Optional[str] = None          # single sequence or colon-separated multi-chain
    sequence_file: Optional[Path] = None    # FASTA file (single or multi-chain)
    yaml_input: Optional[Path] = None       # boltz-compatible YAML (overrides sequence)
    work_dir: Optional[Path] = None         # existing Boltz-2 output directory (rescore)
    receptor_pdb: Optional[Path] = None     # receptor PDB override (rescore only)

    # --- Reference ligand (structure prediction only, non-YAML mode) ---
    # SMILES string or path to SDF file. If omitted, the first ligand in INPUT is used.
    # Used for: Boltz-2 complex prediction + grid center determination.
    reference_ligand: Optional[str] = None

    # --- Grid ---
    grid_center: Optional[Tuple[float, float, float]] = None  # explicit override
    grid_size: float = 20.0

    # --- Ligand chain ID (YAML / rescore modes) ---
    # In non-YAML structure prediction mode, this is derived automatically.
    ligand_chain_id: str = "B"

    # --- Docking ---
    docking_engine: str = "vina"
    num_workers: int = 1
    vina_cpu: int = 1
    batch_size: int = 1
    skip_docking: bool = False
    regenerate_conformer: bool = False

    # --- Boltz-2 prediction params (structure prediction only) ---
    use_msa_server: bool = False
    msa_server_url: str = "https://api.colabfold.com"
    msa_pairing_strategy: str = "greedy"
    msa_server_username: Optional[str] = None
    msa_server_password: Optional[str] = None
    api_key_header: Optional[str] = None
    api_key_value: Optional[str] = None
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    step_scale: Optional[float] = None
    max_parallel_samples: Optional[int] = None
    use_potentials: bool = False
    max_msa_seqs: int = 8192
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024
    no_kernels: bool = False
    affinity_mw_correction: bool = False

    # --- Boltzina scoring params ---
    use_kernels: bool = True
    seed: Optional[int] = None
    keep_intermediate_files: bool = False
    vina_override: bool = False
    boltz_override: bool = False
    float32_matmul_precision: str = "highest"
    run_trunk_and_structure: bool = True

    # --- UniDock2 specific ---
    unidock2_config: Optional[dict] = None

    # --- ligand prefix ---
    ligand_prefix: Optional[str] = None


class BoltzinaRunner:
    """Orchestrates the full Boltzina pipeline."""

    def __init__(self, config: RunnerConfig):
        self.cfg = config
        self._work_dir: Optional[Path] = None
        self._receptor_pdb: Optional[Path] = None
        self._fname: Optional[str] = None
        self._ligand_chain_id: Optional[str] = None  # resolved from YAML or auto-assigned

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return results DataFrame."""
        cfg = self.cfg

        # Validate mode
        has_predict_input = (
            cfg.sequence is not None
            or cfg.sequence_file is not None
            or cfg.yaml_input is not None
        )
        if not has_predict_input and cfg.work_dir is None:
            raise ValueError(
                "Protein input required. Provide one of:\n"
                "  --sequence / --sequence-file  (structure prediction from sequence)\n"
                "  --yaml                        (structure prediction from YAML)\n"
                "  --work-dir                    (rescore from precomputed Boltz-2 results)"
            )

        if cfg.yaml_input is not None:
            self._run_predict_yaml()
        elif has_predict_input:
            sequences = self._resolve_sequences()
            self._run_predict(sequences)
        else:
            self._run_rescore()

        # Prepare ligands
        pdb_paths, pkl_path = self._prepare_ligands()

        # Run Boltzina engine
        results = self._run_boltzina_engine(pdb_paths, pkl_path)
        return results

    # ------------------------------------------------------------------
    # Sequence resolution
    # ------------------------------------------------------------------

    def _resolve_sequences(self) -> list[str]:
        """
        Resolve protein sequences from --sequence or --sequence-file.

        Handles:
          - Single sequence string: "MENFQKV..."
          - Colon-separated multi-chain: "SEQ1:SEQ2"
          - FASTA file (single or multi-chain entries)
        """
        cfg = self.cfg
        if cfg.sequence is not None:
            parts = cfg.sequence.strip().split(":")
            return [p.strip() for p in parts if p.strip()]

        if cfg.sequence_file is not None:
            return _parse_fasta(cfg.sequence_file)

        raise RuntimeError("_resolve_sequences() called without sequence or sequence_file")

    # ------------------------------------------------------------------
    # Pipeline branches
    # ------------------------------------------------------------------

    def _run_predict(self, sequences: list[str]) -> None:
        """Structure prediction pipeline (from sequences)."""
        from boltzina.boltz_setup import setup_boltz_for_run

        cfg = self.cfg
        representative_smiles = self._get_reference_smiles()
        if representative_smiles is None:
            representative_smiles = self._get_first_smiles()
            print(
                f"Using first ligand as reference SMILES for Boltz-2 prediction: "
                f"{representative_smiles[:60]}..."
            )
        else:
            print(f"Reference ligand SMILES: {representative_smiles[:60]}...")

        fname = cfg.output_dir.stem or "boltzina_input"
        work_base_dir = cfg.output_dir / "boltz_work"

        work_dir, receptor_pdb, ligand_chain_id = setup_boltz_for_run(
            sequences=sequences,
            representative_smiles=representative_smiles,
            work_base_dir=work_base_dir,
            fname=fname,
            boltz_kwargs=self._boltz_kwargs(),
        )
        self._work_dir = work_dir
        self._receptor_pdb = receptor_pdb
        self._fname = fname
        self._ligand_chain_id = ligand_chain_id

    def _run_predict_yaml(self) -> None:
        """Structure prediction pipeline (from a pre-written boltz YAML file)."""
        from boltzina.boltz_setup import setup_boltz_for_run_from_yaml

        cfg = self.cfg
        fname = cfg.output_dir.stem or "boltzina_input"
        work_base_dir = cfg.output_dir / "boltz_work"

        work_dir, receptor_pdb, ligand_chain_id = setup_boltz_for_run_from_yaml(
            yaml_path=cfg.yaml_input,
            work_base_dir=work_base_dir,
            fname=fname,
            boltz_kwargs=self._boltz_kwargs(),
        )
        self._work_dir = work_dir
        self._receptor_pdb = receptor_pdb
        self._fname = fname
        self._ligand_chain_id = ligand_chain_id

    def _run_rescore(self) -> None:
        """Rescore pipeline: use existing Boltz-2 results."""
        from boltzina.boltz_setup import extract_receptor_pdb
        import json

        cfg = self.cfg
        self._work_dir = cfg.work_dir
        self._ligand_chain_id = cfg.ligand_chain_id

        # Determine fname from manifest
        manifest_path = cfg.work_dir / "processed" / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"manifest.json not found in {cfg.work_dir / 'processed'}. "
                "Ensure --work-dir points to a valid Boltz-2 output directory."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        self._fname = manifest["records"][0]["id"]

        # Receptor PDB: explicit override or auto-extract
        if cfg.receptor_pdb is not None:
            self._receptor_pdb = cfg.receptor_pdb
        else:
            self._receptor_pdb = extract_receptor_pdb(cfg.work_dir, self._fname)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_reference_smiles(self) -> Optional[str]:
        """
        Return the reference SMILES from --reference-ligand.

        Accepts either a SMILES string or a path to an SDF file.
        Returns None if --reference-ligand was not specified.
        """
        cfg = self.cfg
        if cfg.reference_ligand is None:
            return None

        ref = cfg.reference_ligand.strip()
        ref_path = Path(ref)
        if ref_path.exists():
            # Treat as SDF/PDB file
            from rdkit import Chem
            suffix = ref_path.suffix.lower()
            if suffix in (".sdf", ".mol"):
                supplier = Chem.SDMolSupplier(str(ref_path))
                for mol in supplier:
                    if mol is not None:
                        return Chem.MolToSmiles(mol)
                raise ValueError(f"No molecules found in reference ligand file: {ref_path}")
            elif suffix == ".pdb":
                mol = Chem.MolFromPDBFile(str(ref_path), removeHs=False)
                if mol is not None:
                    return Chem.MolToSmiles(mol)
                raise ValueError(f"Could not read reference ligand PDB: {ref_path}")
            else:
                raise ValueError(
                    f"Unsupported reference ligand format: {suffix}. "
                    "Supported: SMILES string, .sdf, .pdb"
                )
        # Treat as SMILES string
        return ref

    def _get_first_smiles(self) -> str:
        """Extract the first SMILES from the ligand input file."""
        cfg = self.cfg
        input_path = cfg.input_path
        suffix = input_path.suffix.lower()

        if suffix in (".smi", ".txt"):
            with open(input_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line.split()[0]
            raise ValueError(f"No SMILES found in {input_path}")

        elif suffix == ".sdf":
            from rdkit import Chem
            supplier = Chem.SDMolSupplier(str(input_path))
            for mol in supplier:
                if mol is not None:
                    return Chem.MolToSmiles(mol)
            raise ValueError(f"No molecules found in {input_path}")

        raise ValueError(f"Cannot extract SMILES from {input_path}")

    def _boltz_kwargs(self) -> dict:
        """Collect Boltz-2 prediction kwargs from config."""
        cfg = self.cfg
        return dict(
            use_msa_server=cfg.use_msa_server,
            msa_server_url=cfg.msa_server_url,
            msa_pairing_strategy=cfg.msa_pairing_strategy,
            msa_server_username=cfg.msa_server_username,
            msa_server_password=cfg.msa_server_password,
            api_key_header=cfg.api_key_header,
            api_key_value=cfg.api_key_value,
            recycling_steps=cfg.recycling_steps,
            sampling_steps=cfg.sampling_steps,
            diffusion_samples=cfg.diffusion_samples,
            step_scale=cfg.step_scale,
            max_parallel_samples=cfg.max_parallel_samples,
            use_potentials=cfg.use_potentials,
            max_msa_seqs=cfg.max_msa_seqs,
            subsample_msa=cfg.subsample_msa,
            num_subsampled_msa=cfg.num_subsampled_msa,
            no_kernels=cfg.no_kernels,
            affinity_mw_correction=cfg.affinity_mw_correction,
            seed=cfg.seed,
        )

    def _prepare_ligands(self) -> Tuple[list, Path]:
        """Prepare ligands and return (pdb_paths, pkl_path)."""
        cfg = self.cfg
        ligand_prep_dir = cfg.output_dir / "ligands_prepared"
        pdb_paths, pkl_path = prepare_ligands_from_file(
            input_path=cfg.input_path,
            output_dir=ligand_prep_dir,
            ligand_prefix=cfg.ligand_prefix,
            regenerate_conformer=cfg.regenerate_conformer,
        )
        if not pdb_paths:
            raise ValueError(
                f"No ligands were prepared from {cfg.input_path}. "
                "Check that the input file contains at least one valid SMILES or molecule."
            )
        return pdb_paths, pkl_path

    def _run_boltzina_engine(self, pdb_paths: list, pkl_path: Path) -> pd.DataFrame:
        """Run the Boltzina docking+scoring engine."""
        from boltzina.engine import Boltzina

        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolved ligand chain ID (from YAML, auto-assigned, or explicit)
        ligand_chain_id = self._ligand_chain_id or cfg.ligand_chain_id

        # Determine grid center and write Vina config
        vina_config_path = cfg.output_dir / "vina_config.txt"
        center = determine_grid_center(
            work_dir=self._work_dir,
            fname=self._fname,
            ligand_chain_id=ligand_chain_id,
            grid_center=cfg.grid_center,
        )
        write_vina_config(
            center=center,
            output_path=vina_config_path,
            size=cfg.grid_size,
            seed=cfg.seed,
        )

        boltzina = Boltzina(
            receptor_pdb=str(self._receptor_pdb),
            output_dir=str(cfg.output_dir),
            config=str(vina_config_path),
            work_dir=str(self._work_dir),
            fname=self._fname,
            seed=cfg.seed,
            num_workers=cfg.num_workers,
            vina_cpu=cfg.vina_cpu,
            batch_size=cfg.batch_size,
            vina_override=cfg.vina_override,
            boltz_override=cfg.boltz_override,
            use_kernels=cfg.use_kernels,
            skip_docking=cfg.skip_docking,
            clean_intermediate_files=not cfg.keep_intermediate_files,
            run_trunk_and_structure=cfg.run_trunk_and_structure,
            float32_matmul_precision=cfg.float32_matmul_precision,
            ligand_chain_id=ligand_chain_id,
            prepared_mols_file=str(pkl_path),
            docking_engine=cfg.docking_engine,
            unidock2_config=cfg.unidock2_config,
        )

        ligand_files = [str(p) for p in pdb_paths]
        boltzina.run(ligand_files)
        boltzina.save_results_csv()

        df = boltzina.get_results_dataframe()
        print(df.to_string(index=False))
        return df


def _parse_fasta(fasta_path: Path) -> list[str]:
    """
    Parse a FASTA file and return a list of sequences (one per entry).

    Supports standard multi-entry FASTA:
        >entry1
        MENFQKV...
        >entry2
        AKLSILP...

    Returns a list of sequences in the order they appear.
    Raises ValueError if no sequences are found.
    """
    sequences: list[str] = []
    current: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)

    if current:
        sequences.append("".join(current).upper())

    if not sequences:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

    return sequences

"""
BoltzinaRunner — end-to-end orchestrator for `boltzina run`.

Supports two modes:
  Mode A (full-auto): --sequence provided
    1. Generate Boltz-2 YAML from sequence + representative SMILES
    2. Run Boltz-2 structure prediction (or reuse existing)
    3. Extract receptor PDB from Boltz-2 output
    4. Determine grid center from predicted ligand position
    5. Prepare ligands from SMILES/SDF input
    6. Run Boltzina docking + scoring
    7. Save results CSV

  Mode B (existing Boltz results): --work-dir provided
    1. Extract receptor PDB from existing Boltz-2 output (or use --receptor-pdb)
    2. Determine grid center from predicted ligand position (or --grid-center)
    3. Prepare ligands from SMILES/SDF/PDB input
    4. Run Boltzina docking + scoring
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

    # --- Mode A: full-auto ---
    sequence: Optional[str] = None
    sequence_file: Optional[Path] = None
    representative_smiles: Optional[str] = None  # for grid auto in Mode A

    # --- Mode B: existing Boltz results ---
    work_dir: Optional[Path] = None
    receptor_pdb: Optional[Path] = None  # override auto-extraction

    # --- Grid ---
    reference_ligand: Optional[Path] = None  # Mode A: explicit reference ligand
    grid_center: Optional[Tuple[float, float, float]] = None  # explicit override
    grid_size: float = 20.0

    # --- Ligand chain for grid auto (Mode B) ---
    ligand_chain_id: str = "B"

    # --- Docking ---
    docking_engine: str = "vina"
    num_workers: int = 1
    vina_cpu: int = 1
    batch_size: int = 1
    skip_docking: bool = False
    regenerate_conformer: bool = False

    # --- Boltz-2 prediction params (Mode A) ---
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

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline and return results DataFrame.
        """
        cfg = self.cfg

        # Validate mode
        if cfg.sequence is None and cfg.sequence_file is None and cfg.work_dir is None:
            raise ValueError(
                "Either --sequence/--sequence-file (Mode A) or --work-dir (Mode B) is required."
            )

        # Resolve protein sequence for Mode A
        sequence = self._resolve_sequence()

        if sequence is not None:
            self._run_mode_a(sequence)
        else:
            self._run_mode_b()

        # Prepare ligands
        pdb_paths, pkl_path = self._prepare_ligands()

        # Run Boltzina engine
        results = self._run_boltzina_engine(pdb_paths, pkl_path)

        return results

    def _resolve_sequence(self) -> Optional[str]:
        cfg = self.cfg
        if cfg.sequence is not None:
            return cfg.sequence.strip()
        if cfg.sequence_file is not None:
            text = cfg.sequence_file.read_text().strip()
            # Handle FASTA format
            lines = [l for l in text.splitlines() if not l.startswith(">")]
            return "".join(lines).upper()
        return None

    def _run_mode_a(self, sequence: str) -> None:
        """Mode A: run Boltz-2 from scratch."""
        from boltzina.boltz_setup import setup_boltz_for_run

        cfg = self.cfg

        # Determine representative SMILES for Boltz-2 prediction
        representative_smiles = cfg.representative_smiles
        if representative_smiles is None:
            representative_smiles = self._get_first_smiles()
            print(f"Using first ligand as representative SMILES for Boltz-2 prediction: {representative_smiles[:60]}...")

        fname = cfg.output_dir.stem or "boltzina_input"
        work_base_dir = cfg.output_dir / "boltz_work"

        boltz_kwargs = dict(
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

        work_dir, receptor_pdb = setup_boltz_for_run(
            sequence=sequence,
            representative_smiles=representative_smiles,
            work_base_dir=work_base_dir,
            fname=fname,
            ligand_chain_id=cfg.ligand_chain_id,
            boltz_kwargs=boltz_kwargs,
        )
        self._work_dir = work_dir
        self._receptor_pdb = receptor_pdb
        self._fname = fname

    def _run_mode_b(self) -> None:
        """Mode B: use existing Boltz-2 results."""
        from boltzina.boltz_setup import extract_receptor_pdb

        cfg = self.cfg
        self._work_dir = cfg.work_dir

        # Determine fname from manifest
        import json
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

    def _prepare_ligands(self) -> Tuple[list, Path]:
        """Prepare ligands and return (pdb_paths, pkl_path)."""
        cfg = self.cfg
        ligand_prep_dir = cfg.output_dir / "ligands_prepared"
        return prepare_ligands_from_file(
            input_path=cfg.input_path,
            output_dir=ligand_prep_dir,
            ligand_prefix=cfg.ligand_prefix,
            regenerate_conformer=cfg.regenerate_conformer,
        )

    def _get_first_smiles(self) -> str:
        """Extract the first SMILES from the input file."""
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

    def _run_boltzina_engine(self, pdb_paths: list, pkl_path: Path) -> pd.DataFrame:
        """Run the Boltzina docking+scoring engine."""
        from boltzina.engine import Boltzina

        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine grid center and write Vina config
        vina_config_path = cfg.output_dir / "vina_config.txt"
        center = determine_grid_center(
            work_dir=self._work_dir,
            fname=self._fname,
            ligand_chain_id=cfg.ligand_chain_id,
            reference_ligand=cfg.reference_ligand,
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
            ligand_chain_id=cfg.ligand_chain_id,
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

import os
import gc
import subprocess
import pickle
import json
import csv
import copy
import multiprocessing as mp
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem
import shutil
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from boltzina.data.parse.mmcif import parse_mmcif
from boltzina.affinity.predict_affinity import load_boltz2_model, predict_affinity
from boltz.main import get_cache_path


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for multiprocessing.Pool)
# ---------------------------------------------------------------------------

def _process_pose_worker(args):
    """
    Standalone CIF pipeline worker (pdb_chain → pdb_merge → maxit ×2).
    Operates entirely on independent per-ligand files; no shared mutable state.
    """
    (pdb_file, ligand_output_dir, base_name,
     receptor_pdb, ligand_chain_id, input_ligand_name,
     base_ligand_name, vina_override) = args

    pdb_file = Path(pdb_file)
    ligand_output_dir = Path(ligand_output_dir)
    receptor_pdb = Path(receptor_pdb)

    docked_ligands_dir = ligand_output_dir / "docked_ligands"
    docked_ligands_dir.mkdir(parents=True, exist_ok=True)

    prep_file = docked_ligands_dir / f"{base_name}_prep.pdb"
    complex_file = docked_ligands_dir / f"{base_name}_{ligand_chain_id}_complex.pdb"
    complex_cif = docked_ligands_dir / f"{base_name}_{ligand_chain_id}_complex.cif"
    complex_fix_cif = docked_ligands_dir / f"{base_name}_{ligand_chain_id}_complex_fix.cif"

    if complex_fix_cif.exists() and not vina_override:
        return True

    try:
        if input_ligand_name != base_ligand_name:
            cmd1 = (f"pdb_chain -{ligand_chain_id} {pdb_file} | "
                    f"pdb_rplresname -\"{input_ligand_name}\":{base_ligand_name} | "
                    f"pdb_tidy > {prep_file}")
        else:
            cmd1 = f"pdb_chain -{ligand_chain_id} {pdb_file} | pdb_tidy > {prep_file}"
        subprocess.run(cmd1, shell=True, check=True)

        cmd2 = f"pdb_merge {receptor_pdb} {prep_file} | pdb_tidy > {complex_file}"
        subprocess.run(cmd2, shell=True, check=True)

        subprocess.run(
            ["maxit", "-input", str(complex_file), "-output", str(complex_cif), "-o", "1"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["maxit", "-input", str(complex_cif), "-output", str(complex_fix_cif), "-o", "8"],
            check=True, capture_output=True,
        )
        return True
    except Exception as e:
        print(f"  CIF pipeline error for {pdb_file}: {e}")
        return False


# Global state for _prepare_structure_worker (populated by fork initializer)
_g_ccd = None
_g_mol_dict = None
_g_output_dir = None
_g_fname_prefix = None
_g_boltz_override = False


def _prepare_structure_init(ccd, mol_dict, output_dir, fname_prefix, boltz_override):
    global _g_ccd, _g_mol_dict, _g_output_dir, _g_fname_prefix, _g_boltz_override
    _g_ccd = ccd
    _g_mol_dict = mol_dict
    _g_output_dir = output_dir
    _g_fname_prefix = fname_prefix
    _g_boltz_override = boltz_override


def _prepare_structure_worker(args):
    """
    Worker for parse_mmcif. Uses global ccd/mol_dict shared via fork COW.
    """
    complex_file, pose_idx, ligand_idx = args
    complex_file = Path(complex_file)
    output_dir = Path(_g_output_dir)

    fname = f"{_g_fname_prefix}_{ligand_idx}_{pose_idx}"
    pose_output_dir = output_dir / "boltz_out" / "predictions" / fname
    output_path = pose_output_dir / f"pre_affinity_{fname}.npz"

    if output_path.exists() and not _g_boltz_override:
        return fname

    if not complex_file.exists():
        return None

    pose_output_dir.mkdir(parents=True, exist_ok=True)
    extra_mols_dir = output_dir / "out" / str(ligand_idx) / "boltz_out" / "mols"

    try:
        parsed_structure = parse_mmcif(
            path=str(complex_file),
            mols=_g_ccd,
            moldir=extra_mols_dir,
            call_compute_interfaces=False,
        )
        parsed_structure.data.dump(output_path)
        return fname if output_path.exists() else None
    except Exception as e:
        print(f"  parse_mmcif error for {complex_file}: {e}")
        return None

class Boltzina:
    def __init__(
        self,
        receptor_pdb: str,
        output_dir: str,
        config: Optional[str] = None,
        work_dir: Optional[str] = None,
        seed: Optional[int] = None,
        num_workers: int = 4,
        vina_cpu: int = 1,
        batch_size: int = 4,
        num_boltz_poses: int = 1,
        timeout: int = 300,
        vina_override: bool = False,
        boltz_override: bool = False,
        input_ligand_name: str = "MOL",
        base_ligand_name: str = "MOL",
        ligand_chain_id: str = "B",
        fname: Optional[str] = None,
        float32_matmul_precision: str = "highest",
        scoring_only: bool = False,
        skip_docking: bool = False,
        skip_run_structure: bool = True,
        use_kernels: bool = True,
        clean_intermediate_files: bool = True,
        prepared_mols_file: Optional[str] = None,
        predict_affinity_args: Optional[dict] = None,
        pairformer_args: Optional[dict] = None,
        msa_args: Optional[dict] = None,
        steering_args: Optional[dict] = None,
        diffusion_process_args: Optional[dict] = None,
        run_trunk_and_structure: bool = True, # Strongly recommended to be True
        docking_engine: str = "vina",
        unidock2_config: Optional[dict] = None,
    ):
        self.receptor_pdb = Path(receptor_pdb)
        self.output_dir = Path(output_dir)
        self.config = Path(config) if config else None
        self.work_dir = Path(work_dir) if work_dir else None
        self.seed = seed
        self.vina_override = vina_override
        self.boltz_override = boltz_override
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.results = []
        self.num_boltz_poses = num_boltz_poses
        self.pose_idxs = [str(pose_idx) for pose_idx in range(1, self.num_boltz_poses + 1)]
        self.input_ligand_name = input_ligand_name
        self.base_ligand_name = base_ligand_name
        self.ligand_chain_id = ligand_chain_id
        self.float32_matmul_precision = float32_matmul_precision
        self.scoring_only = scoring_only
        self.skip_docking = skip_docking
        self.skip_run_structure = skip_run_structure
        self.use_kernels = use_kernels
        self.timeout = timeout
        self.clean_intermediate_files = clean_intermediate_files
        self.predict_affinity_args = predict_affinity_args
        self.pairformer_args = pairformer_args
        self.msa_args = msa_args
        self.steering_args = steering_args
        self.diffusion_process_args = diffusion_process_args
        self.run_trunk_and_structure = run_trunk_and_structure
        self.docking_engine = docking_engine
        self.unidock2_config = unidock2_config or {}
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_mols_file = prepared_mols_file
        self.mol_dict = None
        self.vina_cpu = vina_cpu

        self.ligand_files = []
        # Prepare receptor PDBQT file (Vina only; Uni-Dock2 handles receptor internally)
        if not self.scoring_only:
            if self.docking_engine == "vina":
                self.receptor_pdbqt = self._prepare_receptor()
            else:
                self.receptor_pdbqt = None
        else:
            self.receptor_pdbqt = self.receptor_pdb

        # Initialize cache directory and CCD
        self.cache_dir = Path(get_cache_path())
        self.ccd_path = self.cache_dir / 'ccd.pkl'
        self.ccd = None
        self._manifest = None  # loaded lazily via self.manifest property

        self.fname = fname  # resolved lazily if None (requires manifest)
        torch.set_float32_matmul_precision(self.float32_matmul_precision)


    def _prepare_receptor(self) -> Path:
        """Prepare receptor PDBQT file using Meeko (mk_prepare_receptor.py)"""
        receptor_pdbqt = self.output_dir / "receptor.pdbqt"
        if receptor_pdbqt.exists() and not self.vina_override:
            print(f"Skipping receptor preparation for {receptor_pdbqt} because it already exists")
            return receptor_pdbqt

        mk_prepare = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
        if mk_prepare is None:
            raise RuntimeError(
                "mk_prepare_receptor.py not found. Install Meeko (e.g., `pip install meeko`) "
                "and ensure your PATH includes the script."
            )

        out_base = str(self.output_dir / "receptor")

        cmd = [
            mk_prepare,
            "-i", str(self.receptor_pdb),
            "-o", out_base,
            "-p",
        ]
        subprocess.run(cmd, check=True)

        produced = Path(out_base + ".pdbqt")
        if not produced.exists():
            produced = receptor_pdbqt

        if not produced.exists():
            raise RuntimeError(f"Failed to create receptor PDBQT file: expected {out_base+'.pdbqt'}")

        if produced != receptor_pdbqt:
            produced.replace(receptor_pdbqt)

        return receptor_pdbqt

    def _load_ccd(self) -> Dict[str, Any]:
        if self.ccd_path.exists():
            with self.ccd_path.open('rb') as file:
                ccd = pickle.load(file)
                if self.base_ligand_name in ccd:
                    ccd.pop(self.base_ligand_name)
                return ccd
        else:
            return {}

    @property
    def manifest(self) -> dict:
        """Lazily load manifest.json from work_dir."""
        if self._manifest is None:
            if self.work_dir is None:
                raise RuntimeError(
                    "work_dir is not set. Provide work_dir to load manifest.json."
                )
            manifest_path = self.work_dir / "processed" / "manifest.json"
            with open(manifest_path, "r") as f:
                self._manifest = json.load(f)
        return self._manifest

    @manifest.setter
    def manifest(self, value: dict) -> None:
        self._manifest = value

    def _get_fname(self) -> str:
        return self.manifest["records"][0]["id"]

    def run(self, ligand_files: List[str]) -> None:
        # Resolve fname lazily here so work_dir can be set after __init__
        if self.fname is None:
            self.fname = self._get_fname()
        if self.scoring_only:
            print("Running scoring only...")
            self.run_scoring_only(ligand_files)
            return
        self.ligand_files = ligand_files
        prep_tasks = []
        for idx, ligand_file in enumerate(ligand_files):
            ligand_path = Path(ligand_file)
            ligand_output_dir = self.output_dir / "out" / str(idx)
            ligand_output_dir.mkdir(parents=True, exist_ok=True)
            if (ligand_output_dir / "done").exists() and not self.vina_override:
                continue
            prep_tasks.append((idx, ligand_path, ligand_output_dir))
        print(f"Docking {len(ligand_files)} ligands with {self.num_workers} workers...")

        if not self.skip_docking:
            if self.docking_engine == "unidock2":
                self._batch_dock_unidock2(prep_tasks)
            elif self.num_workers == 1:
                for task in tqdm(prep_tasks, desc="Preparing ligands"):
                    self._prepare_ligand(task)
            else:
                ligand_num_workers = self.num_workers // self.vina_cpu
                with Pool(ligand_num_workers) as pool:
                    list(tqdm(pool.imap(self._prepare_ligand, prep_tasks), total=len(prep_tasks), desc="Preparing ligands"))
        else:
            print("Skipping docking...")

        self.ccd = self._load_ccd()
        if self.prepared_mols_file:
            with open(self.prepared_mols_file, "rb") as f:
                self.mol_dict = pickle.load(f)

        for idx, ligand_file in enumerate(ligand_files):
            all_exist = True
            for pose_idx in self.pose_idxs:
                fname = f"{self.fname}_{idx}_{pose_idx}"
                if not (self.output_dir / "boltz_out" / "predictions" / fname / f"affinity_{fname}.json").exists():
                    all_exist = False
                    break
            if not all_exist:
                ligand_output_dir = self.output_dir / "out" / str(idx)
                self._update_ccd_for_ligand(ligand_output_dir, Path(ligand_file))

        print("Preparing structures for scoring...")
        structure_tasks = []
        for idx, ligand_file in enumerate(ligand_files):
            ligand_path = Path(ligand_file)
            ligand_output_dir = self.output_dir / "out" / str(idx)
            docked_ligands_dir = ligand_output_dir / "docked_ligands"
            for pose_idx in self.pose_idxs:
                complex_file = docked_ligands_dir / f"docked_ligand_{pose_idx}_{self.ligand_chain_id}_complex_fix.cif"
                fname = f"{self.fname}_{ligand_output_dir.stem}_{pose_idx}"
                affinity_file = self.output_dir / "boltz_out" / "predictions" / fname / f"affinity_{fname}.json"
                pre_affinity_file = self.output_dir / "boltz_out" / "predictions" / fname / f"pre_affinity_{fname}.npz"
                if affinity_file.exists() and not self.boltz_override:
                    continue
                if pre_affinity_file.exists() and not self.boltz_override:
                    continue
                structure_tasks.append((complex_file, pose_idx, idx))

        print(f"Preparing {len(structure_tasks)} structures with {self.num_workers} workers...")
        (self.output_dir / "boltz_out" / "processed").mkdir(parents=True, exist_ok=True)

        max_struct_workers = self.unidock2_config.get("n_pose_workers", os.cpu_count() or 48) if self.docking_engine == "unidock2" else 1
        n_struct_workers = min(max_struct_workers, max(1, len(structure_tasks)))
        worker_tasks = [(str(cf), pi, li) for cf, pi, li in structure_tasks]
        ctx = mp.get_context("fork")
        with ctx.Pool(
            n_struct_workers,
            initializer=_prepare_structure_init,
            initargs=(self.ccd, self.mol_dict,
                      str(self.output_dir), self.fname, self.boltz_override),
        ) as pool:
            list(tqdm(
                pool.imap(_prepare_structure_worker, worker_tasks),
                total=len(worker_tasks), desc="Preparing structures",
            ))

        # clean CCD and mol_dict
        self.ccd = None
        self.mol_dict = None
        gc.collect()

        if self.clean_intermediate_files:
            for complex_file, pose_idx, ligand_idx in tqdm(structure_tasks, desc="Preparing structures"):
                    self._cleanup_preaffinity_intermediates(pose_idx, ligand_idx)

        record_ids = []
        for idx, ligand_file in enumerate(ligand_files):
            ligand_path = Path(ligand_file)
            ligand_output_dir = self.output_dir / "out" / str(idx)
            docked_ligands_dir = ligand_output_dir / "docked_ligands"
            for pose_idx in self.pose_idxs:
                fname = f"{self.fname}_{idx}_{pose_idx}"
                affinity_file = self.output_dir / "boltz_out" / "predictions" / fname / f"affinity_{fname}.json"
                pre_affinity_file = self.output_dir / "boltz_out" / "predictions" / fname / f"pre_affinity_{fname}.npz"
                if affinity_file.exists() and not self.boltz_override:
                    continue
                if not pre_affinity_file.exists(): # If preparation failed
                    continue
                record_ids.append(fname)

        print(f"Affinity prediction will be run for {len(record_ids)} records")
        self._update_manifest(record_ids)
        self._link_constraints(record_ids)

        print("Scoring poses with Boltzina...")
        # Execute scoring with torch multiprocessing
        self._score_poses()
        self._extract_results()

        # Clean up intermediate files after scoring
        if self.clean_intermediate_files:
            self._cleanup_scoring_intermediates()

    def _batch_dock_unidock2(self, prep_tasks: list) -> None:
        """
        Run Uni-Dock2 docking for all ligands in one batch GPU call.

        Steps:
        1. Convert each ligand PDB to SDF (keeping template mols for atom name restoration)
        2. Run Uni-Dock2 once with ligand_batch mode
        3. Split the combined output SDF back to individual PDBs with correct atom names
        4. Run _process_pose() CIF pipeline for each pose
        """
        from boltzina.docking.unidock2 import (
            parse_vina_config,
            pdb_to_sdf,
            run_unidock2_batch,
            split_batch_sdf_to_pdbs,
        )

        if not prep_tasks:
            print("  No ligands to dock (all done).")
            return

        print(f"  Batch docking {len(prep_tasks)} ligands with Uni-Dock2...")

        # Parse center from Vina config
        vina_params = parse_vina_config(self.config)
        center = (
            float(vina_params["center_x"]),
            float(vina_params["center_y"]),
            float(vina_params["center_z"]),
        )

        # Step 1: Convert all PDBs to SDFs
        template_mols = []
        sdf_paths = []
        valid_tasks = []  # tasks where PDB→SDF succeeded

        for task_idx, (idx, ligand_path, ligand_output_dir) in enumerate(prep_tasks):
            ligand_sdf = ligand_output_dir / "ligand.sdf"
            try:
                template_mol = pdb_to_sdf(ligand_path, ligand_sdf)
                # Skip ligands with degenerate 3D coordinates (all atoms collapsed to same position)
                conf = template_mol.GetConformer()
                positions = [conf.GetAtomPosition(i) for i in range(template_mol.GetNumAtoms())]
                if any(
                    positions[i].x == positions[j].x
                    and positions[i].y == positions[j].y
                    and positions[i].z == positions[j].z
                    for i in range(len(positions))
                    for j in range(i + 1, len(positions))
                ):
                    print(f"  Warning: skipping {ligand_path} (idx={idx}): degenerate 3D coordinates")
                    continue
                template_mols.append(template_mol)
                sdf_paths.append(ligand_sdf)
                valid_tasks.append((idx, ligand_path, ligand_output_dir))
            except Exception as e:
                print(f"  Warning: PDB→SDF conversion failed for {ligand_path} (idx={idx}): {e}")

        if not template_mols:
            print("  Error: no ligands could be converted to SDF, aborting batch docking.")
            return

        # Step 2: Run Uni-Dock2 batch
        batch_output_sdf = self.output_dir / "batch_docked.sdf"
        batch_work_dir = self.output_dir / "ud2_batch_work"
        batch_work_dir.mkdir(parents=True, exist_ok=True)

        run_unidock2_batch(
            receptor_pdb=self.receptor_pdb,
            ligand_sdf_list=sdf_paths,
            center=center,
            output_sdf=batch_output_sdf,
            working_dir=batch_work_dir,
            unidock2_config=self.unidock2_config,
        )

        # Step 3: Split combined SDF → per-ligand PDBs with atom names
        output_dirs = [ligand_output_dir for _, _, ligand_output_dir in valid_tasks]
        split_batch_sdf_to_pdbs(
            docked_sdf_path=batch_output_sdf,
            template_mols=template_mols,
            output_dirs=output_dirs,
            num_poses=self.num_boltz_poses,
        )

        # Step 4: CIF pipeline — parallel across all poses (independent files)
        pose_tasks = []
        pose_task_meta = []  # (ligand_path, ligand_output_dir) per pose_task
        for idx, ligand_path, ligand_output_dir in valid_tasks:
            docked_ligands_dir = ligand_output_dir / "docked_ligands"
            for pose_idx_0 in range(self.num_boltz_poses):
                pdb_file = docked_ligands_dir / f"docked_ligand_{pose_idx_0 + 1}.pdb"
                if pdb_file.exists():
                    pose_tasks.append((
                        str(pdb_file),
                        str(ligand_output_dir),
                        f"docked_ligand_{pose_idx_0 + 1}",
                        str(self.receptor_pdb),
                        self.ligand_chain_id,
                        self.input_ligand_name,
                        self.base_ligand_name,
                        self.vina_override,
                    ))
                    pose_task_meta.append((idx, ligand_path, ligand_output_dir))

        max_cif_workers = self.unidock2_config.get("n_pose_workers", os.cpu_count() or 48)
        n_workers = min(max_cif_workers, max(1, len(pose_tasks)))
        print(f"  CIF pipeline: {len(pose_tasks)} poses with {n_workers} workers...")
        if pose_tasks:
            with Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_process_pose_worker, pose_tasks),
                    total=len(pose_tasks), desc="CIF pipeline",
                ))

        # Mark done only if at least one docked PDB was produced for the ligand.
        # Unconditional touch would prevent re-runs from recovering ligands
        # that were skipped due to substructure match failure.
        for idx, ligand_path, ligand_output_dir in valid_tasks:
            docked_ligands_dir = ligand_output_dir / "docked_ligands"
            if any(docked_ligands_dir.glob("docked_ligand_*.pdb")):
                (ligand_output_dir / "done").touch()

        # Cleanup batch intermediates
        if self.clean_intermediate_files:
            self._cleanup_unidock2_batch_intermediates(output_dirs, sdf_paths)

    def _cleanup_unidock2_batch_intermediates(self, output_dirs: list, sdf_paths: list) -> None:
        """Clean up batch-level Uni-Dock2 intermediate files."""
        batch_output_sdf = self.output_dir / "batch_docked.sdf"
        batch_work_dir = self.output_dir / "ud2_batch_work"
        for target in (batch_output_sdf, batch_work_dir):
            try:
                if target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
            except OSError:
                pass
        # Remove per-ligand ligand.sdf files
        for sdf_path in sdf_paths:
            try:
                if Path(sdf_path).is_file():
                    Path(sdf_path).unlink()
            except OSError:
                pass

    def _prepare_ligand(self, task_data):
        """Prepare ligand task for multiprocessing — dispatches to engine-specific method."""
        idx, ligand_path, ligand_output_dir = task_data
        try:
            if self.docking_engine == "unidock2":
                self._prepare_ligand_unidock2(idx, ligand_path, ligand_output_dir)
            else:
                self._prepare_ligand_vina(idx, ligand_path, ligand_output_dir)
            (ligand_output_dir / "done").touch()
        except Exception as e:
            print(f"Error processing ligand {ligand_path} (idx={idx}): {e}")
            done_file = ligand_output_dir / "done"
            if done_file.exists():
                try:
                    done_file.unlink()
                except OSError:
                    pass

    def _prepare_ligand_vina(self, idx: int, ligand_path: Path, ligand_output_dir: Path) -> None:
        """Run Vina docking for a single ligand."""
        ligand_pdbqt = ligand_output_dir / "ligand.pdbqt"
        docked_pdbqt = ligand_output_dir / "docked.pdbqt"
        if not docked_pdbqt.exists():
            self._convert_to_pdbqt(ligand_path, ligand_pdbqt)
            self._run_vina(ligand_pdbqt, docked_pdbqt)
        self._preprocess_docked_structures(idx, docked_pdbqt)
        if self.clean_intermediate_files:
            self._cleanup_vina_intermediates(ligand_output_dir)

    def _prepare_ligand_unidock2(self, idx: int, ligand_path: Path, ligand_output_dir: Path) -> None:
        """Run Uni-Dock2 docking for a single ligand."""
        from boltzina.docking.unidock2 import (
            parse_vina_config,
            pdb_to_sdf,
            run_unidock2,
            split_docked_sdf_to_pdbs,
        )

        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        docked_sdf = ligand_output_dir / "docked.sdf"

        if not docked_sdf.exists() or self.vina_override:
            # Convert input PDB → SDF; template_mol has atom names and same atom order as SDF
            ligand_sdf = ligand_output_dir / "ligand.sdf"
            template_mol = pdb_to_sdf(ligand_path, ligand_sdf)

            # Parse center from Vina config (reuse existing config format)
            vina_params = parse_vina_config(self.config)
            center = (
                float(vina_params["center_x"]),
                float(vina_params["center_y"]),
                float(vina_params["center_z"]),
            )

            ud2_work_dir = ligand_output_dir / "ud2_work"
            ud2_work_dir.mkdir(exist_ok=True)

            run_unidock2(
                receptor_pdb=self.receptor_pdb,
                ligand_sdf=ligand_sdf,
                center=center,
                output_sdf=docked_sdf,
                working_dir=ud2_work_dir,
                unidock2_config=self.unidock2_config,
            )
        else:
            # Docking already done; rebuild template_mol from original PDB for coord mapping
            from rdkit import Chem as _Chem
            template_mol = _Chem.MolFromPDBFile(str(ligand_path), removeHs=True, sanitize=True)
            if template_mol is None:
                template_mol = _Chem.MolFromPDBFile(str(ligand_path), removeHs=True, sanitize=False)

        # Split multi-pose SDF → individual PDBs with correct atom names
        pdb_files = split_docked_sdf_to_pdbs(
            docked_sdf_path=docked_sdf,
            template_mol=template_mol,
            output_dir=docked_ligands_dir,
            num_poses=self.num_boltz_poses,
        )

        # Process each pose through the shared CIF pipeline
        for pose_idx_0, (pdb_file, _score) in enumerate(pdb_files):
            base_name = f"docked_ligand_{pose_idx_0 + 1}"
            self._process_pose(ligand_output_dir, base_name, pdb_file)

        if self.clean_intermediate_files:
            self._cleanup_unidock2_intermediates(ligand_output_dir)

    def _convert_to_pdbqt(self, input_file: Path, output_file: Path) -> None:
        if output_file.exists() and not self.vina_override:
            return
        cmd = ["obabel", str(input_file), "-O", str(output_file)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not output_file.exists():
                raise RuntimeError(f"Failed to create PDBQT file: {output_file}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error converting {input_file} to PDBQT: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during PDBQT conversion: {e}")

    def _run_vina(self, ligand_pdbqt: Path, output_pdbqt: Path) -> None:
        if output_pdbqt.exists() and not self.vina_override:
            print(f"Skipping Vina docking for {output_pdbqt} because it already exists")
            return
        cmd = [
            "vina",
            "--receptor", str(self.receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--out", str(output_pdbqt),
            "--config", str(self.config),
            "--cpu", str(self.vina_cpu),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=self.timeout)
            if not output_pdbqt.exists():
                raise RuntimeError(f"Vina failed to create output file: {output_pdbqt}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Vina docking timed out for {ligand_pdbqt} after {self.timeout} seconds")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Vina docking failed for {ligand_pdbqt}: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Vina docking: {e}")

    def _preprocess_docked_structures(self, ligand_idx: int, docked_pdbqt: Path) -> None:
        ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        docked_ligands_dir.mkdir(exist_ok=True)
        complex_fix_cifs = [
            docked_ligands_dir / f"docked_ligand_{pose_idx}_{self.ligand_chain_id}_complex_fix.cif"
            for pose_idx in self.pose_idxs
        ]

        if all(complex_fix_cif.exists() for complex_fix_cif in complex_fix_cifs) and not self.vina_override:
            return

        # Convert PDBQT to PDB and split into multiple files
        cmd = [
            "obabel", str(docked_pdbqt), "-m", "-O",
            str(docked_ligands_dir / "docked_ligand_.pdb")
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert docked PDBQT to PDB files: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during PDBQT to PDB conversion: {e}")

        # Process each docked pose
        for pose_idx in self.pose_idxs:
            pdb_file = docked_ligands_dir / f"docked_ligand_{pose_idx}.pdb"
            if not pdb_file.exists():
                continue

            ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
            base_name = f"docked_ligand_{pose_idx}"
            self._process_pose(ligand_output_dir, base_name, pdb_file)

    def _process_pose(self, ligand_output_dir: Path, base_name: str, pdb_file: Path) -> None:
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        docked_ligands_dir.mkdir(exist_ok=True)
        prep_file = docked_ligands_dir / f"{base_name}_prep.pdb"
        complex_file = docked_ligands_dir / f"{base_name}_{self.ligand_chain_id}_complex.pdb"
        complex_cif = docked_ligands_dir / f"{base_name}_{self.ligand_chain_id}_complex.cif"
        complex_fix_cif = docked_ligands_dir / f"{base_name}_{self.ligand_chain_id}_complex_fix.cif"
        if complex_fix_cif.exists() and not self.vina_override:
            return
        # Process with pdb_chain and pdb_rplresname
        try:
            if self.input_ligand_name != self.base_ligand_name:
                cmd1 = f"pdb_chain -{self.ligand_chain_id} {pdb_file} | pdb_rplresname -\"{self.input_ligand_name}\":{self.base_ligand_name} | pdb_tidy > {prep_file}"
                subprocess.run(cmd1, shell=True, check=True)
            else:
                cmd1 = f"pdb_chain -{self.ligand_chain_id} {pdb_file} | pdb_tidy > {prep_file}"
                subprocess.run(cmd1, shell=True, check=True)

            if not prep_file.exists():
                raise RuntimeError(f"Failed to create prep file: {prep_file}")

            # Merge with receptor
            cmd2 = f"pdb_merge {self.receptor_pdb} {prep_file} | pdb_tidy > {complex_file}"
            subprocess.run(cmd2, shell=True, check=True)

            if not complex_file.exists():
                raise RuntimeError(f"Failed to create complex file: {complex_file}")

            # Convert to CIF
            cmd3 = [
                "maxit", "-input", str(complex_file), "-output", str(complex_cif), "-o", "1"
            ]
            subprocess.run(cmd3, check=True)

            if not complex_cif.exists():
                raise RuntimeError(f"Failed to create CIF file: {complex_cif}")

            # Fix CIF
            cmd4 = [
                "maxit", "-input", str(complex_cif), "-output", str(complex_fix_cif), "-o", "8"
            ]
            subprocess.run(cmd4, check=True)

            if not complex_fix_cif.exists():
                raise RuntimeError(f"Failed to create fixed CIF file: {complex_fix_cif}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error processing pose {pdb_file}: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error processing pose {pdb_file}: {e}")

    def _update_ccd_for_ligand(self, ligand_output_dir: Path, ligand_path: Optional[Path] = None):
        base_extra_mols_dir = self.output_dir / "boltz_out" / "processed" / "mols"
        base_extra_mols_dir.mkdir(exist_ok=True, parents=True)
        extra_mols_dir = ligand_output_dir / "boltz_out" / "mols"
        extra_mols_dir.mkdir(exist_ok=True, parents=True)

        if (extra_mols_dir / f"{self.base_ligand_name}.pkl").exists() and not self.vina_override:
            all_exist = True
            for pose_idx in self.pose_idxs:
                fname = f"{self.fname}_{ligand_output_dir.stem}_{pose_idx}"
                if not (base_extra_mols_dir / f"{fname}.pkl").exists():
                    all_exist = False
                    break
            if all_exist:
                return

        if self.mol_dict:
            mol = self.mol_dict[ligand_path.stem]
        else:
            mol = Chem.MolFromPDBFile(ligand_path)
            for atom in mol.GetAtoms():
                pdb_info = atom.GetPDBResidueInfo()
                if pdb_info:
                    atom_name = pdb_info.GetName().strip().upper()
                    atom.SetProp("name", atom_name)

        if mol is None:
            raise ValueError(f"Failed to read PDB file {ligand_path}")

        for pose_idx in self.pose_idxs:
            fname = f"{self.fname}_{ligand_output_dir.stem}_{pose_idx}"
            with open(base_extra_mols_dir / f"{fname}.pkl", "wb") as f:
                pickle.dump({self.base_ligand_name: mol}, f)
        with open(extra_mols_dir / f"{self.base_ligand_name}.pkl", "wb") as f:
            pickle.dump(mol, f)
        return

    def _link_constraints(self, record_ids: List[str]) -> None:
        source_constraints_file = self.work_dir / "processed" / "constraints" / f"{self.fname}.npz"
        target_constraints_dir = self.output_dir / "boltz_out" / "processed" / "constraints"
        target_constraints_dir.mkdir(exist_ok=True, parents=True)
        for record_id in record_ids:
            target_constraints_file = target_constraints_dir / f"{record_id}.npz"
            if not target_constraints_file.exists():
                shutil.copy(source_constraints_file, target_constraints_file)
        return

    def _update_manifest(self, record_ids: List[str]) -> None:
        manifest = copy.deepcopy(self.manifest)
        record = [record for record in manifest["records"] if record["id"] == self.fname][0]
        manifest["records"] = []
        for record_id in record_ids:
            new_record = copy.deepcopy(record)
            # for chain_id, _ in enumerate(new_record["chains"]):
            #     if new_record["chains"][chain_id]["msa_id"] != -1:
            #         new_record["chains"][chain_id]["msa_id"] = f"{record_id}_{chain_id}"
            new_record["id"] = record_id
            manifest["records"].append(new_record)
        with open(self.output_dir / "boltz_out" / "processed" / f"manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

    def _prepare_structure_parallel(self, task_data):
        complex_file, pose_idx, ligand_idx = task_data
        return self._prepare_structure(complex_file, pose_idx, ligand_idx)

    def _prepare_structure(self, complex_file: Path, pose_idx: str, ligand_idx: int) -> Optional[Path]:
        """Prepare structure by parsing MMCIF and saving structure data"""
        fname = f"{self.fname}_{ligand_idx}_{pose_idx}"
        pose_output_dir = self.output_dir / "boltz_out" / "predictions" / fname
        if not complex_file.exists():
            try:
                docked_pdbqt = self.output_dir / "out" / str(ligand_idx) / "docked.pdbqt"
                self._preprocess_docked_structures(ligand_idx, docked_pdbqt)
            except Exception as e:
                print(f"Error preparing structure for complex {complex_file} and pose {pose_idx}: {e}")
                return None
        output_path = pose_output_dir / f"pre_affinity_{fname}.npz"
        if output_path.exists() and not self.boltz_override:
            print(f"Skipping structure preparation for pose {pose_idx} because it already exists")
            return pose_output_dir
        pose_output_dir.mkdir(parents=True, exist_ok=True)
        extra_mols_dir = self.output_dir / "out" / str(ligand_idx) / "boltz_out" / "mols"
        try:
            # Parse MMCIF structure
            parsed_structure = parse_mmcif(
                path=str(complex_file),
                mols=self.ccd,
                moldir=extra_mols_dir,
                call_compute_interfaces=False
            )

            # Save structure data
            structure_v2 = parsed_structure.data

            structure_v2.dump(output_path)
            assert output_path.exists(), f"Failed to save structure data for pose {pose_idx} at {output_path}"
            return pose_output_dir

        except Exception as e:
            print(f"Error preparing structure for complex {complex_file} and pose {pose_idx}: {e}")
            return None

    def _score_poses(self):
        """Score a single pose"""
        work_dir = self.work_dir
        output_dir = self.output_dir / "boltz_out" / "predictions"
        extra_mols_dir = self.output_dir / "boltz_out" / "processed" / "mols"
        constraints_dir = self.output_dir / "boltz_out" / "processed" / "constraints"
        # Run Boltzina scoring directly with predict_affinity
        self.boltz_model = load_boltz2_model(skip_run_structure = self.skip_run_structure, use_kernels=self.use_kernels, run_trunk_and_structure=self.run_trunk_and_structure, predict_affinity_args=self.predict_affinity_args, pairformer_args=self.pairformer_args, msa_args=self.msa_args, steering_args=self.steering_args, diffusion_process_args=self.diffusion_process_args)
        predict_affinity(
            work_dir,
            model_module=self.boltz_model,
            output_dir=str(output_dir),  # boltz_out directory
            structures_dir=str(output_dir),
            constraints_dir=str(constraints_dir),
            extra_mols_dir=extra_mols_dir,
            manifest_path = self.output_dir / "boltz_out" / "processed" / "manifest.json",
            num_workers=4,
            batch_size=self.batch_size,
            seed = self.seed,
        )
        self.boltz_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _extract_docking_score(self, docked_pdbqt: Path, pose_idx: int) -> Optional[float]:
        try:
            with open(docked_pdbqt, 'r') as f:
                lines = f.readlines()

            model_count = 0
            for line in lines:
                if line.startswith("MODEL"):
                    if model_count == pose_idx:
                        # Look for REMARK line with score
                        continue
                    model_count += 1
                elif line.startswith("REMARK VINA RESULT:") and model_count == pose_idx:
                    parts = line.split()
                    if len(parts) >= 4:
                        return float(parts[3])
            return None
        except:
            return None

    def _extract_docking_score_unidock2(self, ligand_output_dir: Path, pose_idx: str) -> Optional[float]:
        """Read Uni-Dock2 docking score from the saved JSON file."""
        scores_path = ligand_output_dir / "unidock2_scores.json"
        try:
            with open(scores_path) as f:
                scores = json.load(f)
            return scores.get(str(pose_idx))
        except Exception:
            return None

    def _extract_results(self):
        results = []
        for ligand_idx, ligand_file in enumerate(self.ligand_files):
            for pose_idx in self.pose_idxs:
                fname = f"{self.fname}_{ligand_idx}_{pose_idx}"
                pose_output_dir = self.output_dir / "boltz_out" / "predictions" / fname
                if not (pose_output_dir / f"affinity_{fname}.json").exists():
                    print(f"Skipping {fname} because it doesn't exist")
                    continue
                with open(pose_output_dir / f"affinity_{fname}.json", "r") as f:
                    affinity = json.load(f)
                affinity["ligand_name"] = ligand_file
                affinity["ligand_idx"] = ligand_idx
                affinity["docking_rank"] = pose_idx
                # Set docking score based on mode and engine
                if self.scoring_only:
                    affinity["docking_score"] = None
                elif self.docking_engine == "unidock2":
                    ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
                    affinity["docking_score"] = self._extract_docking_score_unidock2(ligand_output_dir, pose_idx)
                else:
                    affinity["docking_score"] = self._extract_docking_score(self.output_dir / "out" / str(ligand_idx) / "docked.pdbqt", int(pose_idx))
                results.append(affinity)
        self.results = results

    def _cleanup_unidock2_intermediates(self, ligand_output_dir: Path) -> None:
        """Clean up Uni-Dock2 intermediate files, keeping docked.sdf and unidock2_scores.json."""
        for name in ("ligand.sdf", "ud2_work"):
            target = ligand_output_dir / name
            try:
                if target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
            except OSError:
                pass
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        if docked_ligands_dir.exists():
            for file_path in docked_ligands_dir.iterdir():
                keep = (
                    file_path.name.endswith(f"_{self.ligand_chain_id}_complex_fix.cif")
                    or file_path.name == "unidock2_scores.json"
                )
                if not keep:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                    except OSError:
                        pass

    def _cleanup_vina_intermediates(self, ligand_output_dir: Path) -> None:
        """Clean up intermediate files after Vina docking, keeping docked.pdbqt"""
        # Remove ligand.pdbqt
        ligand_pdbqt = ligand_output_dir / "ligand.pdbqt"
        if ligand_pdbqt.exists():
            try:
                ligand_pdbqt.unlink()
            except OSError:
                pass
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        if docked_ligands_dir.exists():
            for file_path in docked_ligands_dir.iterdir():
                if not file_path.name.endswith(f"_{self.ligand_chain_id}_complex_fix.cif"):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)

    def _cleanup_preaffinity_intermediates(self, pose_idx: str, ligand_idx: int) -> None:
        """Clean up intermediate files after pre-affinity calculation"""
        # Check if pre_affinity file exists in predictions directory
        fname = f"{self.fname}_{ligand_idx}_{pose_idx}"
        predictions_dir = self.output_dir / "boltz_out" / "predictions" / fname
        pre_affinity_file = predictions_dir / f"pre_affinity_{fname}.npz"

        if pre_affinity_file.exists():
            # Get the ligand output directory
            ligand_output_dir = self.output_dir / "out" / str(ligand_idx)

            # Remove boltz_out directory under the ligand output directory
            ligand_boltz_out = ligand_output_dir / "boltz_out"
            if ligand_boltz_out.exists():
                try:
                    shutil.rmtree(ligand_boltz_out)
                except OSError:
                    pass

            # Remove docked_ligands directory
            docked_ligands_dir = ligand_output_dir / "docked_ligands"
            if docked_ligands_dir.exists():
                try:
                    shutil.rmtree(docked_ligands_dir)
                except OSError:
                    pass

    def _cleanup_scoring_intermediates(self) -> None:
        """Clean up intermediate files after scoring"""
        boltz_output_dir = self.output_dir / "boltz_out"

        # Remove processed directory and all its contents
        processed_dir = boltz_output_dir / "processed"
        if processed_dir.exists():
            try:
                import shutil
                shutil.rmtree(processed_dir)
            except OSError:
                pass

        # Remove pre_affinity files from predictions directory
        predictions_dir = boltz_output_dir / "predictions"
        if predictions_dir.exists():
            for pre_affinity_file in predictions_dir.glob("*/pre_affinity_*.npz"):
                # Extract the base name to check for corresponding affinity file
                affinity_file = pre_affinity_file.parent / pre_affinity_file.name.replace("pre_affinity_", "affinity_").replace(".npz", ".json")
                if affinity_file.exists():
                    try:
                        pre_affinity_file.unlink()
                    except OSError:
                        pass


    def save_results_csv(self, output_file: Optional[str] = None) -> None:
        if output_file is None:
            output_file = self.output_dir / "boltzina_results.csv"
        else:
            output_file = Path(output_file)

        if not self.results:
            print("No results to save")
            return

        fieldnames = [
            'ligand_name', 'ligand_idx', 'docking_rank', 'docking_score',
            'affinity_pred_value', 'affinity_probability_binary',
            'affinity_pred_value1', 'affinity_probability_binary1',
            'affinity_pred_value2', 'affinity_probability_binary2'
        ]

        # Merge with existing CSV so that incremental re-runs preserve previous results
        existing_rows = []
        existing_keys: set = set()
        if output_file.exists():
            with open(output_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get('ligand_name', ''), row.get('ligand_idx', ''))
                    existing_keys.add(key)
                    existing_rows.append(row)

        new_rows = [
            r for r in self.results
            if (str(r.get('ligand_name', '')), str(r.get('ligand_idx', ''))) not in existing_keys
        ]

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)

        print(f"Results saved to {output_file} ({len(existing_rows)} existing + {len(new_rows)} new)")

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def run_scoring_only(self, ligand_files: List[str]) -> None:
        """
        Run scoring-only mode for ligands with existing poses (no docking).
        Based on scoring_only.py logic.
        """
        # Resolve fname lazily here so work_dir can be set after __init__
        if self.fname is None:
            self.fname = self._get_fname()
        self.ligand_files = ligand_files
        print(f"Running scoring-only mode for {len(self.ligand_files)} ligand poses...")

        if self.ccd is None:
            self.ccd = self._load_ccd()

        if self.prepared_mols_file:
            with open(self.prepared_mols_file, "rb") as f:
                self.mol_dict = pickle.load(f)

        # Process pose files
        for ligand_idx, pdb_file in enumerate(self.ligand_files):
            ligand_path = Path(pdb_file)
            ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
            ligand_output_dir.mkdir(parents=True, exist_ok=True)
            base_name = ligand_path.stem
            self._process_pose(ligand_output_dir, base_name, ligand_path)

        # Update CCD for each ligand
        for ligand_idx, pdb_file in enumerate(self.ligand_files):
            ligand_path = Path(pdb_file)
            ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
            all_exist = True
            for pose_idx in self.pose_idxs:
                fname = f"{self.fname}_{ligand_idx}_{pose_idx}"
                if not (self.output_dir / "boltz_out" / "predictions" / fname / f"affinity_{fname}.json").exists():
                    all_exist = False
                    break
            if not all_exist:
                ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
                self._update_ccd_for_ligand(ligand_output_dir, ligand_path)

        # Process boltz input and prepare structures
        record_ids = []
        for ligand_idx, pdb_file in enumerate(self.ligand_files):
            ligand_path = Path(pdb_file)
            base_name = ligand_path.stem
            ligand_output_dir = self.output_dir / "out" / str(ligand_idx)
            complex_file = ligand_output_dir / "docked_ligands" / f"{base_name}_{self.ligand_chain_id}_complex_fix.cif"

            for pose_idx in self.pose_idxs:
                fname = f"{self.fname}_{ligand_output_dir.stem}_{pose_idx}"
                affinity_file = self.output_dir / "boltz_out" / "predictions" / fname / f"affinity_{fname}.json"
                pre_affinity_file = self.output_dir / "boltz_out" / "predictions" / fname / f"pre_affinity_{fname}.npz"
                if affinity_file.exists() and not self.boltz_override:
                    continue
                self._prepare_structure(complex_file, pose_idx, ligand_idx)
                if not pre_affinity_file.exists():
                    continue
                record_ids.append(fname)

        if self.clean_intermediate_files:
            for ligand_idx, pdb_file in enumerate(self.ligand_files):
                for pose_idx in self.pose_idxs:
                    self._cleanup_preaffinity_intermediates(pose_idx, ligand_idx)

        # Update manifest and link constraints
        self._update_manifest(record_ids)
        self._link_constraints(record_ids)

        # Score poses
        print("Scoring poses with Boltzina...")
        self._score_poses()

        # Extract results
        self._extract_results()

        # Clean up intermediate files after scoring
        if self.clean_intermediate_files:
            self._cleanup_scoring_intermediates()



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Boltzina: Vina docking + Boltz scoring pipeline")
    parser.add_argument("--receptor", required=True, help="Receptor PDB file")
    parser.add_argument("--ligands", required=True, nargs="+", help="Ligand files (SDF/MOL2/SMI)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Vina config file")
    parser.add_argument("--work_dir", help="Working directory for Boltz results")
    parser.add_argument("--vina_override", action="store_true", help="Override existing Vina output directory")
    parser.add_argument("--boltz_override", action="store_true", help="Override existing Boltz output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Boltz parallel processing")
    parser.add_argument("--num_boltz_poses", type=int, default=1, help="Number of Boltz poses to score")
    args = parser.parse_args()

    # Initialize Boltzina
    boltzina = Boltzina(
        receptor_pdb=args.receptor,
        output_dir=args.output_dir,
        config=args.config,
        work_dir=args.work_dir,
        vina_override=args.vina_override,
        boltz_override=args.boltz_override,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_boltz_poses=args.num_boltz_poses
    )

    # Run the pipeline
    boltzina.run(args.ligands)

    # Save results
    boltzina.save_results_csv()

    # Print summary
    df = boltzina.get_results_dataframe()
    print(f"\nProcessed {len(df)} poses from {df['ligand_idx'].nunique()} ligands")
    print(f"Best affinity score: {df['affinity_pred_value'].max():.4f}")


if __name__ == "__main__":
    main()

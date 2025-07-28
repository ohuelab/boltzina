import os
import subprocess
import pickle
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rdkit import Chem
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from boltzina.affinity.process_structure import calc_from_data
from boltzina.affinity.predict_affinity import load_boltz2_model
from boltz.main import get_cache_path


class Boltzina:
    def __init__(self, receptor_pdb: str, output_dir: str, config: str, exhaustiveness: int = 8, mgl_path: Optional[str] = None, work_dir: Optional[str] = None):
        self.receptor_pdb = Path(receptor_pdb)
        self.output_dir = Path(output_dir)
        self.config = Path(config)
        self.exhaustiveness = exhaustiveness
        self.mgl_path = mgl_path
        self.work_dir = work_dir
        self.results = []

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare receptor PDBQT file
        self.receptor_pdbqt = self._prepare_receptor()

        # Initialize cache directory and CCD
        self.cache_dir = Path(get_cache_path())
        self.ccd_path = self.cache_dir / 'ccd.pkl'
        self.ccd = self._load_ccd()

        # Load Boltz2 model once for reuse
        self.boltz_model = load_boltz2_model()

    def _prepare_receptor(self) -> Path:
        """Prepare receptor PDBQT file using prepare_receptor4.py"""
        receptor_pdbqt = self.output_dir / "receptor.pdbqt"

        # Use provided mgl_path or find MGL_PATH (try environment variable or common locations)
        mgl_path = self.mgl_path or os.environ.get('MGL_PATH')
        if not mgl_path:
            # Try common installation paths
            possible_paths = [
                "/usr/local/mgltools",
                "/opt/mgltools",
                os.path.expanduser("~/mgltools")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    mgl_path = path
                    break

        if not mgl_path or not os.path.exists(mgl_path):
            raise RuntimeError("MGL_PATH not found. Please provide mgl_path parameter, set MGL_PATH environment variable, or install MGLTools.")

        prepare_receptor_script = f"{mgl_path}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
        pythonsh = f"{mgl_path}/bin/pythonsh"

        if not os.path.exists(prepare_receptor_script):
            raise RuntimeError(f"prepare_receptor4.py not found at {prepare_receptor_script}")

        # Set up environment
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = f"{mgl_path}/lib"

        # Run prepare_receptor4.py
        cmd = [
            pythonsh,
            prepare_receptor_script,
            "-r", str(self.receptor_pdb),
            "-o", str(receptor_pdbqt)
        ]

        subprocess.run(cmd, env=env, check=True)

        if not receptor_pdbqt.exists():
            raise RuntimeError(f"Failed to create receptor PDBQT file: {receptor_pdbqt}")

        return receptor_pdbqt

    def _load_ccd(self) -> Dict[str, Any]:
        if self.ccd_path.exists():
            with self.ccd_path.open('rb') as file:
                return pickle.load(file)
        else:
            return {}

    def run(self, ligand_files: List[str], ligand_format: str = "sdf") -> None:
        for idx, ligand_file in enumerate(ligand_files):
            ligand_path = Path(ligand_file)

            # Create output directory for this ligand
            ligand_output_dir = self.output_dir / str(idx)
            ligand_output_dir.mkdir(parents=True, exist_ok=True)

            # Convert ligand to PDBQT format if needed
            ligand_pdbqt = ligand_output_dir / "ligand.pdbqt"
            self._convert_to_pdbqt(ligand_path, ligand_pdbqt, ligand_format)

            # Run Vina docking
            docked_pdbqt = ligand_output_dir / "docked.pdbqt"
            self._run_vina(ligand_pdbqt, docked_pdbqt)

            # Preprocess docked structures
            self._preprocess_docked_structures(idx, docked_pdbqt)

            # Update CCD with ligand information
            self._update_ccd_for_ligand(ligand_output_dir)

            # Run Boltzina scoring for each pose
            self._score_poses(idx, ligand_output_dir, ligand_path.stem)

    def _convert_to_pdbqt(self, input_file: Path, output_file: Path, input_format: str) -> None:
        if input_format.lower() in ["sdf", "mol2", "smi"]:
            cmd = ["obabel", str(input_file), "-O", str(output_file)]
            subprocess.run(cmd, check=True)
        else:
            raise ValueError(f"Unsupported ligand format: {input_format}")

    def _run_vina(self, ligand_pdbqt: Path, output_pdbqt: Path) -> None:
        cmd = [
            "vina",
            "--receptor", str(self.receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--out", str(output_pdbqt),
            "--config", str(self.config),
            "--exhaustiveness", str(self.exhaustiveness)
        ]
        subprocess.run(cmd, check=True)

    def _preprocess_docked_structures(self, ligand_idx: int, docked_pdbqt: Path) -> None:
        ligand_output_dir = self.output_dir / str(ligand_idx)
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        docked_ligands_dir.mkdir(exist_ok=True)

        # Convert PDBQT to PDB and split into multiple files
        cmd = [
            "obabel", str(docked_pdbqt), "-m", "-O",
            str(docked_ligands_dir / "docked_ligand_.pdb")
        ]
        subprocess.run(cmd, check=True)

        # Process each docked pose
        for pdb_file in docked_ligands_dir.glob("docked_ligand_*.pdb"):
            if pdb_file.name.endswith("_prep.pdb") or pdb_file.name.endswith("_complex.pdb"):
                continue

            pose_idx = pdb_file.stem.split("_")[-1]
            self._process_pose(ligand_idx, pose_idx, pdb_file)

    def _process_pose(self, ligand_idx: int, pose_idx: str, pdb_file: Path) -> None:
        ligand_output_dir = self.output_dir / str(ligand_idx)
        docked_ligands_dir = ligand_output_dir / "docked_ligands"

        base_name = f"docked_ligand_{pose_idx}"
        prep_file = docked_ligands_dir / f"{base_name}_prep.pdb"
        complex_file = docked_ligands_dir / f"{base_name}_B_complex.pdb"
        complex_cif = docked_ligands_dir / f"{base_name}_B_complex.cif"
        complex_fix_cif = docked_ligands_dir / f"{base_name}_B_complex_fix.cif"

        # Process with pdb_chain and pdb_rplresname
        cmd1 = f"pdb_chain -B {pdb_file} | pdb_rplresname -\"<0>\":MOL | pdb_tidy > {prep_file}"
        subprocess.run(cmd1, shell=True, check=True)

        # Merge with receptor
        cmd2 = f"pdb_merge {self.receptor_pdb} {prep_file} | pdb_tidy > {complex_file}"
        subprocess.run(cmd2, shell=True, check=True)

        # Convert to CIF
        cmd3 = [
            "maxit", "-input", str(complex_file), "-output", str(complex_cif), "-o", "1"
        ]
        subprocess.run(cmd3, check=True)

        # Fix CIF
        cmd4 = [
            "maxit", "-input", str(complex_cif), "-output", str(complex_fix_cif), "-o", "8"
        ]
        subprocess.run(cmd4, check=True)

    def _update_ccd_for_ligand(self, ligand_output_dir: Path, ligand_name: str) -> None:
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        mol_dir = self.cache_dir / "mols"
        mol_dir.mkdir(exist_ok=True, parents=True)
        # Find the first docked ligand PDB file
        pdb_files = list(docked_ligands_dir.glob("docked_ligand_*.pdb"))
        if not pdb_files:
            return

        # Use the first pose to set up CCD
        first_pdb = next(f for f in pdb_files if not f.name.endswith("_prep.pdb") and not f.name.endswith("_complex.pdb"))

        mol = Chem.MolFromPDBFile(str(first_pdb))
        if mol is None:
            return

        for atom in mol.GetAtoms():
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info:
                atom_name = pdb_info.GetName().strip()
                atom.SetProp("name", atom_name)

        with open(mol_dir / f"{ligand_name}.pkl", "wb") as f:
            pickle.dump({"MOL": mol}, f)
        self.ccd["MOL"] = mol

    def _score_poses(self, ligand_idx: int, ligand_output_dir: Path, ligand_name: str) -> None:
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        boltz_output_dir = ligand_output_dir / "boltz_out"
        boltz_output_dir.mkdir(exist_ok=True, parents=True)
        work_dir = self.work_dir or "boltz_results_base_config"

        # Find all processed poses
        complex_files = list(docked_ligands_dir.glob("*_B_complex_fix.cif"))

        for complex_file in complex_files:
            # Extract pose index
            pose_idx = complex_file.stem.split("_")[2]  # docked_ligand_{pose_idx}_B_complex_fix
            fname = f"{ligand_idx}_{pose_idx}"

            try:
                # Run Boltzina scoring and get predictions
                predictions = calc_from_data(
                    cif_path=str(complex_file),
                    output_dir=str(boltz_output_dir),
                    fname=fname,
                    ccd=self.ccd,
                    work_dir=work_dir,
                    model_module=self.boltz_model
                )

                # Extract affinity results from predictions
                if predictions and len(predictions) > 0:
                    pred_data = predictions[0]  # Get the first prediction

                    # Extract affinity values from the prediction data

                    affinity_pred_value = float(pred_data['affinity_pred_value'].item()) if pred_data['affinity_pred_value'] is not None else None

                    affinity_probability_binary = float(pred_data['affinity_probability_binary'].item()) if pred_data['affinity_probability_binary'] is not None else None

                    affinity_pred_value1 = float(pred_data['affinity_pred_value1'].item()) if pred_data['affinity_pred_value1'] is not None else None

                    affinity_probability_binary1 = float(pred_data['affinity_probability_binary1'].item()) if pred_data['affinity_probability_binary1'] is not None else None

                    affinity_pred_value2 = float(pred_data['affinity_pred_value2'].item()) if pred_data['affinity_pred_value2'] is not None else None

                    affinity_probability_binary2 = float(pred_data['affinity_probability_binary2'].item()) if pred_data['affinity_probability_binary2'] is not None else None
                    # Save prediction data as JSON for this pose
                    json_output_dir = boltz_output_dir / "json"
                    json_output_dir.mkdir(exist_ok=True, parents=True)

                    pose_data = {
                        'ligand_name': ligand_name,
                        'ligand_idx': ligand_idx,
                        'docking_rank': int(pose_idx),
                        'docking_score': self._extract_docking_score(ligand_output_dir / "docked.pdbqt", int(pose_idx)),
                        'affinity_pred_value': affinity_pred_value,
                        'affinity_probability_binary': affinity_probability_binary,
                        'affinity_pred_value1': affinity_pred_value1,
                        'affinity_probability_binary1': affinity_probability_binary1,
                        'affinity_pred_value2': affinity_pred_value2,
                        'affinity_probability_binary2': affinity_probability_binary2
                    }

                    json_file = json_output_dir / f"{fname}_prediction.json"
                    with open(json_file, 'w') as f:
                        json.dump(pose_data, f, indent=2, default=str)
                    self.results.append(pose_data)

            except Exception as e:
                print(f"Error scoring pose {pose_idx} for ligand {ligand_idx}: {e}")

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
                elif line.startswith("REMARK VINA RESULT:") and model_count == pose_idx + 1:
                    parts = line.split()
                    if len(parts) >= 4:
                        return float(parts[3])
            return None
        except:
            return None

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

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to {output_file}")

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Boltzina: Vina docking + Boltz scoring pipeline")
    parser.add_argument("--receptor", required=True, help="Receptor PDB file")
    parser.add_argument("--ligands", required=True, nargs="+", help="Ligand files (SDF/MOL2/SMI)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Vina config file")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness parameter")
    parser.add_argument("--ligand_format", default="sdf", help="Ligand file format")
    parser.add_argument("--mgl_path", help="Path to MGLTools installation directory")
    parser.add_argument("--work_dir", help="Working directory for Boltz results")

    args = parser.parse_args()

    # Initialize Boltzina
    boltzina = Boltzina(
        receptor_pdb=args.receptor,
        output_dir=args.output_dir,
        config=args.config,
        exhaustiveness=args.exhaustiveness,
        mgl_path=args.mgl_path,
        work_dir=args.work_dir
    )

    # Run the pipeline
    boltzina.run(args.ligands, args.ligand_format)

    # Save results
    boltzina.save_results_csv()

    # Print summary
    df = boltzina.get_results_dataframe()
    print(f"\nProcessed {len(df)} poses from {df['ligand_idx'].nunique()} ligands")
    print(f"Best affinity score: {df['affinity_pred_value'].max():.4f}")


if __name__ == "__main__":
    main()

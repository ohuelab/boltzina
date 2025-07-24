import os
import subprocess
import pickle
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rdkit import Chem

from boltzina.affinity.process_structure import calc_from_data
from boltz.main import get_cache_path


class Boltzina:
    def __init__(self, receptor_pdbqt: str, output_dir: str, exhaustiveness: int = 8):
        self.receptor_pdbqt = Path(receptor_pdbqt)
        self.output_dir = Path(output_dir)
        self.exhaustiveness = exhaustiveness
        self.results = []

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache directory and CCD
        self.cache_dir = get_cache_path()
        self.ccd_path = self.cache_dir / 'ccd.pkl'
        self.ccd = self._load_ccd()

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
            self._update_ccd_for_ligand(idx, ligand_output_dir)

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
        cmd2 = f"pdb_merge {self.receptor_pdbqt.with_suffix('.pdb')} {prep_file} | pdb_tidy > {complex_file}"
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

    def _update_ccd_for_ligand(self, ligand_idx: int, ligand_output_dir: Path) -> None:
        docked_ligands_dir = ligand_output_dir / "docked_ligands"

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

        self.ccd["MOL"] = mol

    def _score_poses(self, ligand_idx: int, ligand_output_dir: Path, ligand_name: str) -> None:
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        boltz_output_dir = ligand_output_dir / "boltz_out"
        work_dir = "test_data/KIF11/boltz_out/boltz_results_base_config"

        # Find all processed poses
        complex_files = list(docked_ligands_dir.glob("*_B_complex_fix.cif"))

        for complex_file in complex_files:
            # Extract pose index
            pose_idx = complex_file.stem.split("_")[2]  # docked_ligand_{pose_idx}_B_complex_fix
            fname = f"{ligand_idx}_{pose_idx}"

            try:
                # Run Boltzina scoring
                result = calc_from_data(
                    cif_path=str(complex_file),
                    output_dir=str(boltz_output_dir),
                    fname=fname,
                    ccd=self.ccd,
                    work_dir=work_dir
                )

                # Parse affinity results
                affinity_file = boltz_output_dir / f"{fname}" / f"affinity_{fname}.json"
                if affinity_file.exists():
                    with open(affinity_file, 'r') as f:
                        affinity_data = json.load(f)

                    self.results.append({
                        'ligand_name': ligand_name,
                        'ligand_idx': ligand_idx,
                        'docking_rank': int(pose_idx),
                        'docking_score': self._extract_docking_score(ligand_output_dir / "docked.pdbqt", int(pose_idx)),
                        'affinity_pred_value': affinity_data.get('affinity_pred_value', None),
                        'affinity_probability_binary': affinity_data.get('affinity_probability_binary', None),
                        'affinity_pred_value1': affinity_data.get('affinity_pred_value1', None),
                        'affinity_probability_binary1': affinity_data.get('affinity_probability_binary1', None),
                        'affinity_pred_value2': affinity_data.get('affinity_pred_value2', None),
                        'affinity_probability_binary2': affinity_data.get('affinity_probability_binary2', None)
                    })

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
    parser.add_argument("--receptor", required=True, help="Receptor PDBQT file")
    parser.add_argument("--ligands", required=True, nargs="+", help="Ligand files (SDF/MOL2/SMI)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness parameter")
    parser.add_argument("--ligand_format", default="sdf", help="Ligand file format")

    args = parser.parse_args()

    # Initialize Boltzina
    boltzina = Boltzina(
        receptor_pdbqt=args.receptor,
        output_dir=args.output_dir,
        exhaustiveness=args.exhaustiveness
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

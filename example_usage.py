#!/usr/bin/env python3

from boltzina_main import Boltzina
from pathlib import Path

def example_usage():
    # Example usage of Boltzina
    receptor_pdbqt = "test_data/KIF11/docking/receptor.pdbqt"
    ligand_files = ["test_data/KIF11/active_mols/CHEMBL1163892.sdf"]
    output_dir = "example_output"

    # Check if test files exist
    if not Path(receptor_pdbqt).exists():
        print(f"Receptor file {receptor_pdbqt} not found. Please provide a valid receptor PDBQT file.")
        return

    if not all(Path(f).exists() for f in ligand_files):
        print("Some ligand files not found. Please provide valid ligand files.")
        return

    print("Initializing Boltzina...")
    boltzina = Boltzina(
        receptor_pdbqt=receptor_pdbqt,
        output_dir=output_dir,
        exhaustiveness=8
    )

    print("Running Vina docking and Boltz scoring...")
    try:
        boltzina.run(ligand_files, ligand_format="sdf")

        print("Saving results to CSV...")
        boltzina.save_results_csv()

        # Display results summary
        df = boltzina.get_results_dataframe()
        if not df.empty:
            print(f"\nResults Summary:")
            print(f"Total poses processed: {len(df)}")
            print(f"Ligands processed: {df['ligand_idx'].nunique()}")
            print(f"Best docking score: {df['docking_score'].min():.3f}")
            print(f"Best affinity prediction: {df['affinity_pred_value'].max():.3f}")

            print(f"\nTop 5 poses by affinity prediction:")
            top_poses = df.nlargest(5, 'affinity_pred_value')[
                ['ligand_name', 'docking_rank', 'docking_score', 'affinity_pred_value']
            ]
            print(top_poses.to_string(index=False))
        else:
            print("No results generated.")

    except Exception as e:
        print(f"Error during processing: {e}")
        print("Make sure all required tools are installed:")
        print("- AutoDock Vina")
        print("- OpenBabel")
        print("- PDB tools")
        print("- maxit")

if __name__ == "__main__":
    example_usage()

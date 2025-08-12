#!/usr/bin/env python3
import json
import argparse
from boltzina_main import Boltzina
from pathlib import Path

MGL_PATH = "/PATH/TO/mgltools_x86_64Linux2_1.5.7"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for Boltz-2 Scoring, batch_size = 1 is strongly recommended")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for AutoDock Vina")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generator")
    parser.add_argument("--vina_override", action="store_true", help="Override results of AutoDock Vina")
    parser.add_argument("--boltz_override", action="store_true", help="Override results of Boltz-2 Scoring")
    parser.add_argument("--use_kernels", action="store_true", help="Use Boltz-2 kernels for scoring")
    parser.add_argument("--skip_docking", action="store_true", help="Skip docking")
    parser.add_argument("--float32_matmul_precision", type=str, default="highest", choices=["highest", "high", "medium"], help="Precision for float32 matmul")
    parser.add_argument("--skip_trunk_and_structure", action="store_true", help="Skip running trunk and structure")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    receptor_pdb = config_dict["receptor_pdb"]
    ligand_files = config_dict["ligand_files"]
    output_dir = args.output_dir if args.output_dir else config_dict["output_dir"]
    work_dir = config_dict["work_dir"]
    config = config_dict["vina_config"]
    input_ligand_name = config_dict["input_ligand_name"]
    fname = config_dict["fname"]
    seed = args.seed if config_dict.get("seed", None) is None else config_dict["seed"]
    float32_matmul_precision = config_dict.get("float32_matmul_precision", args.float32_matmul_precision)
    scoring_only = config_dict.get("scoring_only", False)
    prepared_mols_file = config_dict.get("prepared_mols_file", None)
    predict_affinity_args = config_dict.get("predict_affinity_args", None)
    pairformer_args = config_dict.get("pairformer_args", None)
    msa_args = config_dict.get("msa_args", None)
    steering_args = config_dict.get("steering_args", None)
    diffusion_process_args = config_dict.get("diffusion_process_args", None)
    run_trunk_and_structure = not args.skip_trunk_and_structure
    print("--------------------------------")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {seed}")
    print(f"Mode: {'scoring only' if scoring_only else 'docking'}")
    print(f"Using float32 matmul precision: {float32_matmul_precision}")

    boltzina = Boltzina(
        receptor_pdb=receptor_pdb,
        output_dir=output_dir,
        config=config,
        mgl_path=MGL_PATH,
        work_dir=work_dir,
        input_ligand_name=input_ligand_name,
        fname=fname,
        seed=seed,
        vina_override=args.vina_override,
        boltz_override=args.boltz_override,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        float32_matmul_precision=float32_matmul_precision,
        scoring_only=scoring_only,
        prepared_mols_file=prepared_mols_file,
        use_kernels=args.use_kernels,
        predict_affinity_args=predict_affinity_args,
        pairformer_args=pairformer_args,
        msa_args=msa_args,
        steering_args=steering_args,
        diffusion_process_args=diffusion_process_args,
        skip_docking = args.skip_docking,
        run_trunk_and_structure = run_trunk_and_structure
    )

    boltzina.run(ligand_files)

    print("Saving results to CSV...")
    boltzina.save_results_csv()

    df = boltzina.get_results_dataframe()
    print(df)

if __name__ == "__main__":
    main()

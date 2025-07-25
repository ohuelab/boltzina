# Boltzina

Boltzina is a pipeline that combines AutoDock Vina docking with Boltz scoring for molecular docking and affinity prediction.

## Usage

```bash
python boltzina_main.py \
    --receptor test_data/KIF11/docking/receptor.pdbqt \
    --ligands test_data/KIF11/active_mols/CHEMBL1163892.sdf \
    --output_dir example_output \
    --config test_data/KIF11/docking/input.txt \
    --exhaustiveness 8 \
    --ligand_format sdf
```

## Parameters

- `--receptor`: Receptor PDBQT file
- `--ligands`: Ligand files (SDF/MOL2/SMI format)
- `--output_dir`: Output directory for results
- `--config`: Vina config file (required - contains binding site coordinates)
- `--exhaustiveness`: Vina exhaustiveness parameter (default: 8)
- `--ligand_format`: Ligand file format (default: sdf)

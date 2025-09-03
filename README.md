# Boltzina
![png](https://arxiv.org/html/2508.17555v1/x1.png)
Boltzina is a pipeline that combines AutoDock Vina docking with Boltz-2 scoring for molecular docking and affinity prediction.

## Installation

```bash
# Using uv (recommended)
uv venv
uv sync

# Or using pip
pip install .
```

Related environments including AutoDock Vina, Maxit, and Boltz-2 model checkpoint files can be installed with the following command:
```bash
./setup.sh
```


## Modes

Boltzina supports two operation modes:

1. **Full docking mode**: Performs AutoDock Vina docking followed by Boltz-2 scoring
2. **Scoring-only mode**: Scores pre-existing ligand poses using only Boltz-2 (no docking)

## Usage

Run the pipeline using a configuration file:

```bash
python run.py sample/CDK2/config.json
```

Example for scoring mode:
```bash
python run.py sample/CDK2/config_scoring.json
```

## Ligand File Format
To generate a Boltzina-compatible input PDB file and a mols_dict pkl from SMILES, follow the steps below:
```bash
# Prepare ligand PDB Files
$ python ligand_preparation.py preparation_sample/input_smiles.txt --output_dir preparation_sample
# Run Boltzina
$ python run.py preparation_sample/config.json
```

For input file formats, please refer to `INPUT_FORMAT.md`.

## Command Line Options

- `config`: Path to configuration JSON file (required)
- `--batch_size`: Batch size for Boltz-2 scoring (default: 1, strongly recommended)
- `--num_workers`: Number of workers for AutoDock Vina (default: 1)
- `--vina_override`: Override existing AutoDock Vina results
- `--boltz_override`: Override existing Boltz-2 scoring results

## Configuration File Format

The configuration file should be a JSON file with the following required fields:

```json
{
    "work_dir": "sample/CDK2/boltz_results_base",
    "vina_config": "sample/CDK2/input.txt",
    "fname": "1ckp_cdk2",
    "input_ligand_name": "UNL",
    "output_dir": "sample/CDK2/results",
    "receptor_pdb": "sample/CDK2/boltz_results_base/predictions/1ckp_cdk2/1ckp_cdk2_model_0_protein.pdb",
    "ligand_files": [
        "sample/CDK2/input_pdbs/CDK2_active_0.pdb",
        "sample/CDK2/input_pdbs/CDK2_active_1.pdb"
    ]
}
```

### Configuration Parameters

- **`work_dir`**: Working directory for intermediate files
- **`vina_config`**: Path to AutoDock Vina configuration file (contains binding site coordinates)
- **`fname`**: Base filename for output files
- **`input_ligand_name`**: Name of the ligand in the input files
- **`output_dir`**: Directory where final results will be saved
- **`receptor_pdb`**: Path to the receptor PDB file
- **`ligand_files`**: Array of paths to ligand files (Now only supports PDB format)

## Reference
Furui K, Ohue M. Boltzina: Efficient and Accurate Virtual Screening via Docking-Guided Binding Prediction with Boltz-2, arXiv (preprint) 2025. https://doi.org/10.48550/arXiv.2508.17555

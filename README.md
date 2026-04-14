# Boltzina
![png](https://arxiv.org/html/2508.17555v1/x1.png)
Boltzina is a pipeline that combines AutoDock Vina (or Uni-Dock2) docking with Boltz-2 scoring for molecular docking and affinity prediction.

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv venv
uv sync

# Or using pip
pip install .
```

### Tool setup (Vina, MAXIT, Boltz-2 model weights)

```bash
boltzina setup --all
```

For Uni-Dock2 (GPU-accelerated docking):
```bash
boltzina setup --install-unidock2   # shows installation guide
boltzina setup --register-unidock2 /path/to/Uni-Dock2  # after install
```

---

## Usage

### Mode A — Full auto (protein sequence + ligand list)

Provide a protein sequence and a SMILES/SDF file. Boltzina will:
1. Run Boltz-2 structure + affinity prediction
2. Determine the docking grid automatically from the predicted binding pose
3. Run AutoDock Vina docking
4. Score all poses with Boltz-2

```bash
boltzina run ligands.smi \
  --sequence "MENFQKVEKIGEGTYGVVYK..." \
  --output-dir ./results
```

With a FASTA file:
```bash
boltzina run ligands.smi \
  --sequence-file protein.fasta \
  --output-dir ./results \
  --use-msa-server \          # use online MSA server
  --diffusion-samples 5       # more samples for better accuracy
```

With an SDF file (3D coordinates preserved automatically):
```bash
boltzina run ligands.sdf \
  --sequence "MENFQKV..." \
  --output-dir ./results
```

### Mode B — Existing Boltz-2 results

If you have already run `boltz predict`, pass the output directory directly:

```bash
boltzina run ligands.smi \
  --work-dir ./boltz_results_1ckp_cdk2 \
  --output-dir ./results
```

The grid center is determined automatically from the Boltz-2 predicted ligand position.
You can override it explicitly:

```bash
boltzina run ligands.smi \
  --work-dir ./boltz_results_1ckp_cdk2 \
  --grid-center "7.0,-4.9,7.5" \
  --output-dir ./results
```

### Sample run (CDK2)

```bash
boltzina run sample/CDK2/input.txt \
  --work-dir sample/CDK2/boltz_results_base \
  --output-dir ./cdk2_results
```

---

## CLI Reference

### `boltzina run <INPUT> [OPTIONS]`

`INPUT` can be a `.smi`/`.txt` file (SMILES list), `.sdf` file, or a directory.

| Option | Default | Description |
|--------|---------|-------------|
| `--sequence` / `-s` | — | Protein amino acid sequence (Mode A) |
| `--sequence-file` | — | FASTA/text file with protein sequence (Mode A) |
| `--work-dir` | — | Existing Boltz-2 output directory (Mode B) |
| `--output-dir` / `-o` | `./boltzina_results` | Output directory |
| `--grid-center` | auto | Docking box center `x,y,z` |
| `--grid-size` | `20.0` | Docking box size (Å) |
| `--ligand-chain-id` | `B` | Ligand chain in Boltz-2 prediction |
| `--docking-engine` | `vina` | `vina` or `unidock2` |
| `--num-workers` | `1` | Parallel Vina workers |
| `--batch-size` | `1` | Boltz-2 scoring batch size |
| `--skip-docking` | off | Score existing poses only |
| `--regenerate-conformer` | off | Force 3D conformer regeneration for SDF |
| `--no-kernels` | off | Disable trifast kernels (older GPUs) |
| `--seed` | — | Random seed |
| `--use-msa-server` | off | Use online MMseqs2 MSA server |
| `--recycling-steps` | `3` | Boltz-2 recycling steps |
| `--sampling-steps` | `200` | Boltz-2 sampling steps |
| `--diffusion-samples` | `1` | Boltz-2 diffusion samples |
| `--use-potentials` | off | Boltz-2 inference-time potentials |
| `--affinity-mw-correction` | off | MW correction to affinity |
| `--vina-override` | off | Rerun Vina even if results exist |
| `--boltz-override` | off | Rerun Boltz-2 scoring even if results exist |
| `--keep-intermediate-files` | off | Keep intermediate docking files |

### `boltzina prepare <INPUT> [OPTIONS]`

Convert SMILES/SDF to PDB + `prepared_mols.pkl` for use with `run.py`.

```bash
boltzina prepare ligands.smi --output-dir ./prepared
boltzina prepare ligands.sdf --output-dir ./prepared --regenerate-conformer
```

### `boltzina grid <STRUCTURE_FILE> [OPTIONS]`

Compute the docking grid center from a ligand or complex file.

```bash
boltzina grid ligand.pdb --output vina_config.txt
boltzina grid complex.cif --chain B --output vina_config.txt
```

### `boltzina setup [OPTIONS]`

Install and register external tools.

```bash
boltzina setup --all                          # Vina + MAXIT + Boltz-2 weights
boltzina setup --install-vina                 # Vina only
boltzina setup --install-maxit                # MAXIT only
boltzina setup --install-unidock2             # Show Uni-Dock2 install guide
boltzina setup --register-unidock2 /path/to/Uni-Dock2
boltzina setup --show                         # Show current config
```

---

## Legacy usage (run.py)

The original `run.py` interface is fully supported:

```bash
python run.py sample/CDK2/config.json
python run.py sample/CDK2/config.json --use_kernels --num_workers 4
```

See `sample/CDK2/config.json` for the configuration file format.

---

## Running Tests

```bash
uv run pytest tests/ -m gpu
```

Integration tests (full pipeline, requires GPU + Boltz-2 weights):
```bash
uv run pytest tests/test_integration.py -m "gpu" -v
```

---

## Reference
Furui, K, & Ohue, M. Boltzina: Efficient and Accurate Virtual Screening via Docking-Guided Binding Prediction with Boltz-2. AI for Accelerated Materials Design - NeurIPS 2025. https://openreview.net/forum?id=OwtEQsd2hN

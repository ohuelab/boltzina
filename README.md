# Boltzina
![png](https://arxiv.org/html/2508.17555v1/x1.png)
Boltzina is a pipeline that combines AutoDock Vina (or Uni-Dock2) docking with Boltz-2 structure prediction and affinity scoring for virtual screening.

## Quick Start

### Installation

```bash
# From PyPI
pip install boltzina

# From source (development)
uv sync
```

### Tool setup (Vina, MAXIT, Boltz-2 model weights)

```bash
boltzina setup --all
```

For Uni-Dock2 (GPU-accelerated docking, requires [pixi](https://pixi.sh) and CUDA 12):
```bash
# Clone Uni-Dock2 and build using the provided pixi.toml
git clone https://github.com/dptech-corp/Uni-Dock2 /path/to/Uni-Dock2
cp pixi.toml /path/to/Uni-Dock2/
cd /path/to/Uni-Dock2 && pixi install && pixi run build
boltzina setup --register-unidock2 /path/to/Uni-Dock2
```

---

## Usage

### With Boltz-2 structure prediction (sequence → dock → score)

Provide a protein sequence and a SMILES/SDF file. Boltzina will:
1. Run Boltz-2 structure + affinity prediction (complex with first/reference ligand)
2. Determine the docking grid automatically from the predicted binding pose
3. Run AutoDock Vina docking
4. Score all poses with Boltz-2

```bash
# From a FASTA file (CDK2 example)
boltzina run sample/CDK2/ligands.smi \
  --sequence-file sample/CDK2/cdk2.fasta \
  --output-dir ./results

# From a sequence string directly
boltzina run sample/CDK2/ligands.smi \
  --sequence "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL" \
  --output-dir ./results

# Multi-chain protein: colon-separated sequences
boltzina run sample/CDK2/ligands.smi \
  --sequence "MENFQKVEKIGEGTYGVVYK...:AKLSILPWGHC..." \
  --output-dir ./results

# Multi-chain protein: multi-entry FASTA
boltzina run sample/CDK2/ligands.smi \
  --sequence-file complex.fasta \   # >chain1 / seq / >chain2 / seq
  --output-dir ./results

# Use a specific reference ligand for prediction and grid center
boltzina run sample/CDK2/ligands.smi \
  --sequence-file sample/CDK2/cdk2.fasta \
  --reference-ligand "CC(C)[C@H](CO)Nc1nc(Nc2ccc(C(=O)O)c(Cl)c2)c2ncn(C(C)C)c2n1" \
  --output-dir ./results

# With more diffusion samples for better accuracy
boltzina run sample/CDK2/ligands.smi \
  --sequence-file sample/CDK2/cdk2.fasta \
  --use-msa-server \
  --diffusion-samples 5 \
  --output-dir ./results
```

### With a Boltz-2 YAML input

For full control over multi-chain proteins, ligand definitions, and Boltz-2 settings,
use a boltz-compatible YAML file (see `sample/CDK2/1ckp_cdk2.yaml` for an example):

```bash
boltzina run sample/CDK2/ligands.smi \
  --yaml sample/CDK2/1ckp_cdk2.yaml \
  --output-dir ./results
```

The YAML format:
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MENFQKVEKIGEGTYGVVYK...  # CDK2 sequence
  - ligand:
      id: B
      smiles: 'CC(C)[C@H](CO)Nc1nc(Nc2ccc(C(=O)O)c(Cl)c2)c2ncn(C(C)C)c2n1'
properties:
  - affinity:
      binder: B
```

Multiple protein chains are supported (add more `- protein:` entries).
The `properties.affinity.binder` identifies the reference ligand for grid center determination.

### From precomputed Boltz-2 results

If you have already run `boltz predict`, pass the output directory directly:

```bash
boltzina run sample/CDK2/ligands.smi \
  --work-dir sample/CDK2/boltz_results_base \
  --output-dir ./results
```

The grid center is determined automatically from the Boltz-2 predicted ligand position.
You can override it explicitly:

```bash
boltzina run sample/CDK2/ligands.smi \
  --work-dir sample/CDK2/boltz_results_base \
  --grid-center "7.0,-4.9,7.5" \
  --output-dir ./results
```

---

## CLI Reference

### `boltzina run <INPUT> [OPTIONS]`

`INPUT` can be a `.smi`/`.txt` file (SMILES list), `.sdf` file, or a directory.

**Protein input** (choose one; required):

| Option | Description |
|--------|-------------|
| `--sequence` / `-s` | Protein sequence (single chain, or `SEQ1:SEQ2` for multi-chain) |
| `--sequence-file` | FASTA file (one `>entry` per chain for multi-chain) |
| `--yaml` | Boltz-2 compatible YAML (protein + ligand + affinity) |
| `--work-dir` | Existing Boltz-2 output directory (docking + scoring only) |

**Structure prediction options** (with `--sequence` / `--sequence-file`):

| Option | Default | Description |
|--------|---------|-------------|
| `--reference-ligand` | first in INPUT | SMILES string or SDF file for Boltz-2 complex prediction and grid center |

**Docking:**

| Option | Default | Description |
|--------|---------|-------------|
| `--grid-center` | auto | Docking box center `x,y,z` |
| `--grid-size` | `20.0` | Docking box size (Å) |
| `--ligand-chain-id` | `B` | Ligand chain in Boltz-2 prediction (rescore mode) |
| `--docking-engine` | `vina` | `vina` or `unidock2` |
| `--num-workers` | `1` | Parallel Vina workers |
| `--skip-docking` | off | Score existing poses only |
| `--regenerate-conformer` | off | Force 3D conformer regeneration for SDF |

**Boltz-2 prediction:**

| Option | Default | Description |
|--------|---------|-------------|
| `--use-msa-server` | off | Use online MMseqs2 MSA server |
| `--recycling-steps` | `3` | Boltz-2 recycling steps |
| `--sampling-steps` | `200` | Boltz-2 sampling steps |
| `--diffusion-samples` | `1` | Boltz-2 diffusion samples |
| `--use-potentials` | off | Boltz-2 inference-time potentials |
| `--subsample-msa` | off | Subsample MSA sequences |
| `--no-kernels` | off | Disable trifast kernels (older GPUs) |
| `--affinity-mw-correction` | off | MW correction to affinity |

**Output:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` / `-o` | `./boltzina_results` | Output directory |
| `--batch-size` | `1` | Boltz-2 scoring batch size |
| `--seed` | — | Random seed |
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

## Benchmark Dataset

The MF-PCBA benchmark dataset used in the paper is included in `mf-pcba_test.zip`.
See the paper for details on the evaluation protocol.

---

## Running Tests

```bash
# Unit tests (no GPU required)
uv run pytest tests/ --ignore=tests/test_integration.py -v

# Integration tests (requires GPU + Boltz-2 weights)
uv run pytest tests/test_integration.py -m gpu -v
```

---

## Reference
Furui, K, & Ohue, M. Boltzina: Efficient and Accurate Virtual Screening via Docking-Guided Binding Prediction with Boltz-2. AI for Accelerated Materials Design - NeurIPS 2025. https://openreview.net/forum?id=OwtEQsd2hN

# BioIR affinity scoring

Use `--scoring-backend bioir` to score docked poses with NVIDIA's
[BioNeMo Inference Runtime (BioIR)](https://docs.nvidia.com/bionemo/inference-runtime/latest/overview/).
Boltzina supplies the prepared features and fixed coordinates directly to BioIR
in the same Python process. Both scoring backends use the Boltz-2 affinity
checkpoint and write the same result columns. Docking is selected independently
with `--docking-engine vina` or `--docking-engine unidock2`.

## Python environment

The supplied environment targets Linux, Python 3.12, and NVIDIA Ampere or newer
GPUs. The BioIR release requires glibc 2.34 or newer and NVIDIA driver 580 or
newer. Available GPU memory determines the usable batch size and input size.
See the [BioIR support matrix](https://docs.nvidia.com/bionemo/inference-runtime/latest/references/support-matrix/).

From a Boltzina source checkout, install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --project environments/bioir --locked --no-dev
source environments/bioir/.venv/bin/activate
boltzina setup --all
export RCSBROOT="$HOME/.boltzina/maxit-v11.300-prod-src"
boltzina run ligands.smi --work-dir boltz_results \
  --scoring-backend bioir --batch-size 4 --seed 42 --output-dir results
```

The environment pins BioIR 0.2.1 and PyTorch 2.11.0. It explicitly overrides
Boltz's pins for Triton, Biopython, ModelCIF, and Gemmi, and replaces the CUDA 12
cuEquivariance operators with CUDA 13 operators and cuEquivariance 0.13.0.
These choices are recorded in
`environments/bioir/pyproject.toml` and `uv.lock`. Use that environment rather
than adding BioIR to an existing Boltz environment with pip.

The usual sequence, FASTA, YAML, and precomputed-input modes are available.
`--scoring-backend` selects fixed-pose affinity scoring; the initial structure
prediction stage still uses Boltz. `--no-kernels` is available with the default
`boltz` scoring backend.

## Containers

The container recipes build the same Python environment and include AutoDock
Vina, MAXIT, and Open Babel. The recipes target Linux x86_64. Model weights and
input data are supplied at runtime. Uni-Dock2 can be registered separately with
`boltzina setup --register-unidock2` in an environment where its executable and
dependencies are accessible.

The `/cache/boltz` mount also stores runtime and compilation caches and must be
writable.

### Docker

Build from the repository root:

```bash
docker build -f containers/Dockerfile.bioir -t boltzina-bioir .
mkdir -p cache results
docker run --rm --gpus all \
  -v "$PWD/cache:/cache/boltz" boltzina-bioir setup --download-models
docker run --rm --gpus all \
  -v "$PWD:/data" -v "$PWD/cache:/cache/boltz" \
  boltzina-bioir run ligands.smi --work-dir boltz_results \
  --scoring-backend bioir --batch-size 4 --seed 42 --output-dir results
```

### Apptainer

Build from the repository root on a machine configured for container builds:

```bash
apptainer build --fakeroot boltzina-bioir.sif containers/bioir.def
```

The resulting image can be copied to another compatible GPU machine. Run with
GPU access and writable directories for data, results, and the model cache:

```bash
mkdir -p cache results
if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
apptainer exec --nv --cleanenv --bind "$PWD/cache:/cache/boltz" \
  boltzina-bioir.sif boltzina setup --download-models
apptainer exec --nv --cleanenv \
  --bind "$PWD:/data" --bind "$PWD/cache:/cache/boltz" --pwd /data \
  boltzina-bioir.sif boltzina run ligands.smi --work-dir boltz_results \
  --scoring-backend bioir --batch-size 4 --seed 42 --output-dir results
```

The conditional export preserves any GPU assignment made by a scheduler when
using `--cleanenv`.

## Results and reuse

Boltzina writes `scoring_backend.json` in the output directory and BioIR runtime
details in `boltz_out/scoring_metadata.json`. The latter records the checkpoint
hash, software versions, seed, batch size, and valid/failed record counts.

Use separate output directories to compare backends. To replace existing scores
with another backend in the same directory, pass `--boltz-override` explicitly.
Boltzina checks dependencies before starting preparation and does not silently
switch scoring implementations.

## Validation

Run the tests in the BioIR environment:

```bash
uv sync --project environments/bioir --locked
uv run --project environments/bioir --locked python -m pytest -m 'not gpu'
uv run --project environments/bioir --locked python -m pytest \
  tests/test_bioir_model.py -m gpu
```

The GPU test requires cached model weights and configured Vina/MAXIT tools. It
runs docking, feature generation, two-record affinity scoring, and result saving
using the bundled protein example.

# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.1] - 2026-11-08

Boltzina 1.1.1 fixes a CUDA Graph capture error that could stop fixed-pose
affinity scoring with `--scoring-backend bioir` when using BioIR 0.2.1.

### Fixed

- Disable automatic CUDA Graph capture in the BioIR trunk so that CPU-based MSA
  sampling runs on each recycle without being captured in a CUDA graph.

## [1.1.0] - 2026-11-07

Boltzina 1.1.0 adds accelerated fixed-pose affinity scoring with NVIDIA BioNeMo
Inference Runtime (BioIR). The BioIR backend can score fixed poses **about 2×
faster** than Boltzina's standard backend.

Select it with `--scoring-backend bioir` using the supplied Python environment or
Docker/Apptainer recipes. See the [BioIR installation and usage guide](https://github.com/ohuelab/boltzina/blob/v1.1.0/docs/bioir.md).

### Added
- Optional BioIR backend for accelerated fixed-pose affinity scoring (`--scoring-backend bioir`). The default backend remains `boltz`.
- A reproducible BioIR Python environment and Docker/Apptainer recipes with Vina, MAXIT, and Open Babel.

### Changed
- Reuse parsed protein MSAs across ligands to reduce repeated data loading.

### Fixed
- Pin compatible cuEquivariance packages for the standard Boltz environment.
- Clear previous affinity results and runtime metadata when explicitly switching scoring backends with `--boltz-override`.
- Handle receptor CIF-to-PDB conversion when ligand component names exceed the PDB format limit.
- Report the correct `boltzina_results.csv` path after a successful run.

## [1.0.1] - 2026-05-16

### Fixed
- Fixed fresh Boltz-2 prediction from `--yaml` and `--sequence` inputs with recent Boltz versions, where calling the Click command object directly raised `TypeError: Context.__init__() got an unexpected keyword argument 'data'`.
- Added offline MSA handling for generated and user-provided YAML inputs by setting missing protein MSA entries to `empty` unless `--use-msa-server` is requested.
- Improved external tool discovery for Vina, MAXIT, Open Babel, Meeko, and pdb-tools by resolving executables from the active Python environment as well as configured paths and `PATH`.
- Avoided the ProDy-dependent Meeko path for protein-only PDB receptor preparation.
- Normalized docked ligand residue names from `UNL` to `MOL` before Boltzina scoring so ligands are not dropped by mmCIF parsing.
- Replaced recursive scoring fallback with invalid-record prefiltering so one affinity crop failure does not recurse into record 0.
- Rebuilt ligand molecule metadata when rerunning with `--boltz-override`.

## [1.0.0] - 2026-04-21

### Added
- UniDock2 docking backend alongside existing Vina (`--docking_engine unidock2`)
- Multi-chain receptor support
- YAML input mode for per-ligand metadata
- `--reference-ligand` to auto-define docking box from a reference PDB
- `--mask_ligand_coords` for NoPose ablation benchmarks
- `boltzina setup --all` for one-command dependency setup
- Full test suite (73 tests)

### Changed
- Rewritten as a structured installable package (`boltzina.engine`, `boltzina.runner`, etc.)
- Startup time reduced from ~8 s to <1 s (lazy imports)

### Removed
- `boltzina_main.py` (replaced by `boltzina.engine.Boltzina`)
- `setup.sh`, `example_usage.py`, `ligand_preparation.py`, `INPUT_FORMAT.md`

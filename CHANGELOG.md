# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

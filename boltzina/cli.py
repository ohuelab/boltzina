"""
Boltzina CLI entry point.

Usage:
    boltzina run <INPUT> [OPTIONS]       # Main pipeline
    boltzina prepare <INPUT> [OPTIONS]   # SMILES/SDF → PDB + pkl
    boltzina grid <STRUCT> [OPTIONS]     # Determine docking grid center
    boltzina setup [OPTIONS]             # Install/register tools
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup, MutuallyExclusiveOptionGroup

from boltzina import __version__


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="boltzina")
def main():
    """Boltzina: virtual screening via docking-guided Boltz-2 affinity prediction."""
    pass


# ---------------------------------------------------------------------------
# boltzina run
# ---------------------------------------------------------------------------

@main.command("run")
@click.argument("input_path", type=click.Path(exists=True))

# Protein input group (mutually exclusive; one required)
@optgroup.group("Protein input", cls=RequiredMutuallyExclusiveOptionGroup,
                help="Specify the protein target (choose one)")
@optgroup.option("--sequence", "-s", default=None,
                 help="Protein sequence (single chain or colon-separated multi-chain: 'SEQ1:SEQ2')")
@optgroup.option("--sequence-file", type=click.Path(exists=True), default=None,
                 help="FASTA file (single or multi-chain; each >entry = one protein chain)")
@optgroup.option("--yaml", type=click.Path(exists=True), default=None,
                 help="Boltz-2 compatible YAML (protein + ligand + affinity property)")
@optgroup.option("--work-dir", type=click.Path(exists=True), default=None,
                 help="Precomputed Boltz-2 output directory (docking + scoring only)")

# Reference ligand (structure prediction only)
@optgroup.group("Structure prediction options",
                help="Options for --sequence / --sequence-file (ignored with --yaml or --work-dir)")
@optgroup.option("--reference-ligand", default=None,
                 help="Reference ligand: SMILES string or SDF/PDB file. "
                      "Used for Boltz-2 complex prediction and grid center. "
                      "Default: first ligand in INPUT.")
@optgroup.option("--receptor-pdb", type=click.Path(exists=True), default=None,
                 help="Receptor PDB override (rescore mode only; auto-extracted from --work-dir if omitted)")

# Docking group
@optgroup.group("Docking")
@optgroup.option("--docking-engine", default="vina", show_default=True,
                 type=click.Choice(["vina", "unidock2"]),
                 help="Docking engine")
@optgroup.option("--grid-center", default=None,
                 help="Explicit grid center 'x,y,z' (overrides auto-detection)")
@optgroup.option("--grid-size", default=20.0, show_default=True,
                 help="Docking box size in Angstroms")
@optgroup.option("--ligand-chain-id", default="B", show_default=True,
                 help="Ligand chain ID in Boltz-2 prediction (for grid auto in --work-dir mode; "
                      "auto-assigned in sequence mode)")
@optgroup.option("--num-workers", default=1, show_default=True,
                 help="Parallel workers for Vina docking")
@optgroup.option("--vina-cpu", default=1, show_default=True,
                 help="CPUs per Vina worker")
@optgroup.option("--skip-docking", is_flag=True,
                 help="Skip docking; score existing poses only")
@optgroup.option("--regenerate-conformer", is_flag=True,
                 help="Force 3D conformer regeneration for SDF inputs")

# Boltz-2 prediction group
@optgroup.group("Boltz-2 prediction",
                help="Parameters for Boltz-2 structure prediction (ignored with --work-dir)")
@optgroup.option("--use-msa-server", is_flag=True,
                 help="Use mmseqs2 MSA server (requires internet)")
@optgroup.option("--msa-server-url", default="https://api.colabfold.com", show_default=True,
                 help="MSA server URL")
@optgroup.option("--msa-server-username", default=None, help="MSA server username")
@optgroup.option("--msa-server-password", default=None, help="MSA server password")
@optgroup.option("--msa-pairing-strategy", default="greedy", show_default=True,
                 type=click.Choice(["greedy", "complete"]),
                 help="MSA pairing strategy")
@optgroup.option("--recycling-steps", default=3, show_default=True,
                 help="Boltz-2 recycling steps")
@optgroup.option("--sampling-steps", default=200, show_default=True,
                 help="Boltz-2 sampling steps")
@optgroup.option("--diffusion-samples", default=1, show_default=True,
                 help="Boltz-2 diffusion samples")
@optgroup.option("--step-scale", default=None, type=float,
                 help="Boltz-2 step scale (default: 1.638)")
@optgroup.option("--max-parallel-samples", default=None, type=int,
                 help="Max parallel diffusion samples")
@optgroup.option("--subsample-msa", is_flag=True,
                 help="Subsample MSA sequences")
@optgroup.option("--num-subsampled-msa", default=1024, show_default=True,
                 help="Number of subsampled MSA sequences (used with --subsample-msa)")
@optgroup.option("--use-potentials", is_flag=True,
                 help="Use Boltz-2 inference-time potentials")
@optgroup.option("--max-msa-seqs", default=8192, show_default=True,
                 help="Maximum MSA sequences")
@optgroup.option("--no-kernels", is_flag=True,
                 help="Disable trifast kernels (use on older GPUs)")
@optgroup.option("--affinity-mw-correction", is_flag=True,
                 help="Apply molecular weight correction to affinity")

# Output / misc group
@optgroup.group("Output")
@optgroup.option("--output-dir", "-o", default="./boltzina_results", show_default=True,
                 help="Output directory")
@optgroup.option("--batch-size", default=1, show_default=True,
                 help="Batch size for Boltz-2 affinity scoring")
@optgroup.option("--seed", default=None, type=int, help="Random seed")
@optgroup.option("--keep-intermediate-files", is_flag=True,
                 help="Keep intermediate docking/CIF files")
@optgroup.option("--float32-matmul-precision", default="highest",
                 type=click.Choice(["highest", "high", "medium"]),
                 help="PyTorch float32 matmul precision")
@optgroup.option("--vina-override", is_flag=True,
                 help="Rerun Vina even if results exist")
@optgroup.option("--boltz-override", is_flag=True,
                 help="Rerun Boltz-2 scoring even if results exist")
@optgroup.option("--ligand-prefix", default=None,
                 help="Prefix for ligand names (fallback when SMILES file has no name column)")
def run_cmd(
    input_path,
    sequence, sequence_file, yaml, work_dir,
    reference_ligand, receptor_pdb,
    docking_engine, grid_center, grid_size, ligand_chain_id,
    num_workers, vina_cpu, skip_docking, regenerate_conformer,
    use_msa_server, msa_server_url, msa_server_username, msa_server_password,
    msa_pairing_strategy, recycling_steps, sampling_steps, diffusion_samples,
    step_scale, max_parallel_samples, subsample_msa, num_subsampled_msa,
    use_potentials, max_msa_seqs, no_kernels, affinity_mw_correction,
    output_dir, batch_size, seed, keep_intermediate_files, float32_matmul_precision,
    vina_override, boltz_override, ligand_prefix,
):
    """
    Run the Boltzina virtual screening pipeline.

    INPUT: Ligand file (.smi, .sdf) or directory containing ligand files.

    \b
    Structure prediction:
      boltzina run ligands.smi --sequence-file protein.fasta --output-dir ./results
      boltzina run ligands.smi --yaml input.yaml --output-dir ./results

    \b
    From precomputed Boltz-2 results:
      boltzina run ligands.smi --work-dir ./boltz_results --output-dir ./results
    """
    from boltzina.runner import BoltzinaRunner, RunnerConfig

    # Parse grid center
    grid_center_tuple = None
    if grid_center is not None:
        try:
            parts = [float(x.strip()) for x in grid_center.split(",")]
            if len(parts) != 3:
                raise ValueError
            grid_center_tuple = tuple(parts)
        except (ValueError, AttributeError):
            raise click.BadParameter(
                f"--grid-center must be 'x,y,z' (e.g. '7.0,-4.9,7.5'), got: {grid_center}"
            )

    # Exclusivity checks for --yaml and --work-dir
    if yaml is not None:
        if reference_ligand is not None:
            raise click.UsageError(
                "--yaml already contains the reference ligand. "
                "Do not combine with --reference-ligand."
            )
        if ligand_chain_id != "B":
            raise click.UsageError(
                "--ligand-chain-id is derived automatically from --yaml "
                "(properties.affinity.binder). Do not specify it explicitly."
            )

    if work_dir is not None:
        if reference_ligand is not None:
            raise click.UsageError(
                "--reference-ligand is only valid for structure prediction "
                "(--sequence / --sequence-file / --yaml), not for --work-dir."
            )

    # Echo pipeline description
    click.echo(f"Boltzina v{__version__}")
    if yaml is not None:
        click.echo("Docking with Boltz-2 structure prediction (from YAML)")
    elif sequence is not None or sequence_file is not None:
        click.echo("Docking with Boltz-2 structure prediction (from sequence)")
    else:
        click.echo("Docking from precomputed Boltz-2 results")
    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_dir}")

    cfg = RunnerConfig(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        sequence=sequence,
        sequence_file=Path(sequence_file) if sequence_file else None,
        yaml_input=Path(yaml) if yaml else None,
        work_dir=Path(work_dir) if work_dir else None,
        receptor_pdb=Path(receptor_pdb) if receptor_pdb else None,
        reference_ligand=reference_ligand,
        grid_center=grid_center_tuple,
        grid_size=grid_size,
        ligand_chain_id=ligand_chain_id,
        docking_engine=docking_engine,
        num_workers=num_workers,
        vina_cpu=vina_cpu,
        batch_size=batch_size,
        skip_docking=skip_docking,
        regenerate_conformer=regenerate_conformer,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        msa_pairing_strategy=msa_pairing_strategy,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        step_scale=step_scale,
        max_parallel_samples=max_parallel_samples,
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_potentials=use_potentials,
        max_msa_seqs=max_msa_seqs,
        no_kernels=no_kernels,
        use_kernels=not no_kernels,
        affinity_mw_correction=affinity_mw_correction,
        seed=seed,
        keep_intermediate_files=keep_intermediate_files,
        float32_matmul_precision=float32_matmul_precision,
        vina_override=vina_override,
        boltz_override=boltz_override,
        ligand_prefix=ligand_prefix,
    )

    runner = BoltzinaRunner(cfg)
    df = runner.run()
    click.echo(f"\nDone. Results saved to {Path(output_dir) / 'results.csv'}")
    if not df.empty:
        best = df.loc[df["affinity_pred_value"].idxmin()] if "affinity_pred_value" in df.columns else None
        if best is not None:
            click.echo(f"Best ligand: {best.get('ligand_name', '?')} "
                       f"(affinity_pred_value={best['affinity_pred_value']:.3f})")


# ---------------------------------------------------------------------------
# boltzina prepare
# ---------------------------------------------------------------------------

@main.command("prepare")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="./prepared_ligands", show_default=True,
              help="Output directory")
@click.option("--ligand-prefix", default=None,
              help="Prefix for ligand names")
@click.option("--regenerate-conformer", is_flag=True,
              help="Force 3D conformer regeneration for SDF inputs")
def prepare_cmd(input_path, output_dir, ligand_prefix, regenerate_conformer):
    """
    Prepare ligands from a SMILES (.smi) or SDF (.sdf) file.

    Generates PDB files with unique atom names and a prepared_mols.pkl
    for consistent Boltz-2 scoring.
    """
    from boltzina.preparation import prepare_ligands_from_file

    pdb_paths, pkl_path = prepare_ligands_from_file(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        ligand_prefix=ligand_prefix,
        regenerate_conformer=regenerate_conformer,
    )
    click.echo(f"Prepared {len(pdb_paths)} ligands → {Path(output_dir) / 'input_pdbs'}")
    click.echo(f"PKL: {pkl_path}")


# ---------------------------------------------------------------------------
# boltzina grid
# ---------------------------------------------------------------------------

@main.command("grid")
@click.argument("structure_file", type=click.Path(exists=True))
@click.option("--chain", default=None,
              help="Chain ID to compute center for (CIF only)")
@click.option("--output", "-o", default="vina_config.txt", show_default=True,
              help="Output Vina config file path")
@click.option("--size", default=20.0, show_default=True,
              help="Box size in Angstroms")
def grid_cmd(structure_file, chain, output, size):
    """
    Compute docking grid center from a ligand or structure file.

    STRUCTURE_FILE can be a ligand PDB/SDF/MOL2 (center of mass)
    or a complex CIF (extracts specified chain center).
    """
    import numpy as np
    from boltzina.docking.grid import (
        determine_grid_center, write_vina_config,
        _parse_cif_ligand_center, _get_center_from_pdb_chain,
    )
    from boltzina.docking.calculate_com import get_center_of_mass_from_file

    p = Path(structure_file)
    if p.suffix == ".cif" and chain:
        center = _parse_cif_ligand_center(p, chain)
    elif p.suffix == ".pdb" and chain:
        center = _get_center_from_pdb_chain(p, chain)
    else:
        center = get_center_of_mass_from_file(str(p))

    write_vina_config(center=center, output_path=Path(output), size=size)
    click.echo(f"Config written to {output}")


# ---------------------------------------------------------------------------
# boltzina setup
# ---------------------------------------------------------------------------

@main.command("setup")
@click.option("--install-vina", is_flag=True, help="Download and install AutoDock Vina")
@click.option("--install-maxit", is_flag=True, help="Download and build MAXIT")
@click.option("--register-unidock2", default=None, metavar="PATH",
              help="Register installed Uni-Dock2 repo path in config")
@click.option("--register-vina", default=None, metavar="PATH",
              help="Register an existing Vina binary path")
@click.option("--register-maxit", default=None, metavar="PATH",
              help="Register an existing MAXIT binary path")
@click.option("--download-models", is_flag=True,
              help="Download Boltz-2 model checkpoints")
@click.option("--all", "all_tools", is_flag=True,
              help="Install Vina + MAXIT + download Boltz-2 models")
@click.option("--show", is_flag=True, help="Show current tool configuration")
def setup_cmd(
    install_vina, install_maxit,
    register_unidock2, register_vina, register_maxit,
    download_models, all_tools, show,
):
    """
    Install and configure external tools required by Boltzina.

    Tools:
      - AutoDock Vina (auto-downloaded binary)
      - MAXIT (source build)
      - Uni-Dock2 (requires manual build; register with --register-unidock2)
      - Boltz-2 models (downloaded from HuggingFace)
    """
    from boltzina.config import (
        _load_config,
        register_vina as _reg_vina,
        register_maxit as _reg_maxit,
        register_unidock2 as _reg_ud2,
        show_config,
    )

    if show:
        show_config()
        return

    if register_vina:
        _reg_vina(register_vina)
        return

    if register_maxit:
        _reg_maxit(register_maxit)
        return

    if register_unidock2:
        existing = _load_config().get("tools", {}).get("unidock2_env", {}).get("bin")
        if existing:
            click.confirm(
                f"Uni-Dock2 is already registered at '{existing}'. Overwrite?",
                abort=True,
            )
        _reg_ud2(register_unidock2)
        return

    if all_tools:
        install_vina = True
        install_maxit = True
        download_models = True

    if install_vina:
        _install_vina()

    if install_maxit:
        _install_maxit()

    if download_models:
        _download_boltz_models()

    if not any([install_vina, install_maxit, register_unidock2,
                register_vina, register_maxit, download_models, all_tools, show]):
        click.echo("No action specified. Run 'boltzina setup --help' for options.")


def _install_vina():
    """Download and install AutoDock Vina binary."""
    import subprocess
    from boltzina.config import CONFIG_DIR, register_vina

    bin_dir = CONFIG_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    vina_path = bin_dir / "vina"

    if vina_path.exists():
        click.echo(f"Vina already exists at {vina_path}")
        register_vina(str(vina_path))
        return

    url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64"
    click.echo(f"Downloading AutoDock Vina from {url} ...")
    result = subprocess.run(
        ["wget", "-q", url, "-O", str(vina_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo(f"Download failed: {result.stderr}", err=True)
        sys.exit(1)

    vina_path.chmod(0o755)
    register_vina(str(vina_path))
    click.echo("AutoDock Vina installed successfully.")


def _install_maxit():
    """Download and build MAXIT."""
    import subprocess
    from boltzina.config import CONFIG_DIR, register_maxit

    maxit_dir = CONFIG_DIR / "maxit-v11.300-prod-src"
    maxit_bin = maxit_dir / "bin" / "maxit"

    if maxit_bin.exists():
        click.echo(f"MAXIT already built at {maxit_bin}")
        register_maxit(str(maxit_bin))
        return

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tarball = CONFIG_DIR / "maxit-v11.300-prod-src.tar.gz"
    url = "https://sw-tools.rcsb.org/apps/MAXIT/maxit-v11.300-prod-src.tar.gz"

    if not tarball.exists():
        click.echo(f"Downloading MAXIT from {url} ...")
        result = subprocess.run(
            ["wget", "-q", url, "-O", str(tarball)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            click.echo(f"Download failed: {result.stderr}", err=True)
            sys.exit(1)

    click.echo("Extracting MAXIT ...")
    subprocess.run(["tar", "xf", str(tarball), "-C", str(CONFIG_DIR)], check=True)

    click.echo("Building MAXIT (this may take a few minutes) ...")
    result = subprocess.run(
        ["make", "binary"],
        cwd=str(maxit_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not maxit_bin.exists():
        click.echo(f"Build failed:\n{result.stdout}\n{result.stderr}", err=True)
        sys.exit(1)

    register_maxit(str(maxit_bin))
    click.echo("MAXIT built and registered successfully.")
    click.echo(f"Add to PATH: export RCSBROOT={maxit_dir}")


def _download_boltz_models():
    """Download Boltz-2 model checkpoints."""
    from boltzina.config import get_boltz_cache

    cache = get_boltz_cache()
    click.echo(f"Downloading Boltz-2 models to {cache} ...")
    from boltz.main import download_boltz1, download_boltz2
    download_boltz1(cache)
    download_boltz2(cache)
    click.echo("Boltz-2 models downloaded.")

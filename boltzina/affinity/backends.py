"""Validation and provenance for affinity scoring backends."""

import importlib.util
import json
from importlib.metadata import version
from pathlib import Path


def validate_scoring_backend(backend: str, *, use_kernels: bool = True) -> None:
    if backend not in ("boltz", "bioir"):
        raise ValueError(f"Unknown scoring backend: {backend!r}. Choose 'boltz' or 'bioir'.")
    if backend == "boltz":
        return
    if not use_kernels:
        raise ValueError("--no-kernels cannot be combined with --scoring-backend bioir.")
    if importlib.util.find_spec("bionemo_ir") is None:
        raise RuntimeError(
            "BioIR is not installed. Use the environment described in docs/bioir.md "
            "to run with --scoring-backend bioir."
        )
    if version("bionemo-ir") != "0.2.1":
        raise RuntimeError("BioIR scoring requires bionemo-ir==0.2.1. See docs/bioir.md.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BioIR scoring requires an available NVIDIA CUDA GPU.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("BioIR scoring requires an NVIDIA Ampere or newer GPU.")


def check_cached_backend(output_dir: Path, backend: str, *, override: bool) -> None:
    """Prevent reuse of affinity files produced by a different backend."""
    if override:
        return
    marker = output_dir / "scoring_backend.json"
    metadata = json.loads(marker.read_text()) if marker.exists() else {"scoring_backend": "boltz"}
    previous = metadata["scoring_backend"]
    has_results = any((output_dir / "boltz_out/predictions").glob("*/affinity_*.json"))
    if metadata.get("state") == "running" and has_results:
        raise ValueError("Previous affinity scoring did not finish. Use --boltz-override or another output directory.")
    if previous != backend and has_results:
        raise ValueError(
            f"Existing affinity results use {previous!r}, but {backend!r} was requested. "
            "Choose another output directory or use --boltz-override to rescore."
        )


def write_backend_metadata(output_dir: Path, backend: str, **settings) -> None:
    marker = output_dir / "scoring_backend.json"
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"scoring_backend": backend, **settings}, indent=2) + "\n")
    temporary.replace(marker)

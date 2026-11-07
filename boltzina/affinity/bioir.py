"""Run BioIR scoring with Boltzina's data loader and result writer."""

import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch

from boltzina.affinity.backends import validate_scoring_backend


def predict_bioir_affinity(
    out_dir, *, output_dir, constraints_dir, extra_mols_dir, manifest_path,
    batch_size=1, num_workers=4, seed=None, recycling_steps=5,
):
    validate_scoring_backend("bioir")
    from boltz.main import get_cache_path
    from boltz.data.types import Manifest
    from pytorch_lightning import seed_everything
    from boltzina.affinity.bioir_model import FixedPoseAffinity
    from boltzina.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltzina.data.write.writer import BoltzAffinityWriter

    if batch_size < 1 or num_workers < 0 or recycling_steps < 0:
        raise ValueError("Invalid batch size, worker count, or recycling count.")
    if seed is not None:
        seed_everything(seed)
    cache = Path(get_cache_path())
    checkpoint = cache / "boltz2_aff.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Affinity checkpoint not found: {checkpoint}. Run boltzina setup --all.")
    output_dir = Path(output_dir)
    data = Boltz2InferenceDataModule(
        manifest=Manifest.load(Path(manifest_path)), target_dir=output_dir,
        msa_dir=Path(out_dir) / "processed/msa", mol_dir=cache / "mols",
        constraints_dir=Path(constraints_dir), extra_mols_dir=Path(extra_mols_dir),
        num_workers=num_workers, override_method="other", affinity=True, batch_size=batch_size,
    )
    dataset = data.predict_dataset()
    if not len(dataset):
        raise RuntimeError("No valid affinity records to score after prefiltering.")
    writer = BoltzAffinityWriter(data_dir=output_dir, output_dir=output_dir)
    model = FixedPoseAffinity.from_checkpoint(checkpoint).cuda()
    completed = 0
    try:
        for index, batch in enumerate(data.predict_dataloader()):
            features = {
                key: value.cuda(non_blocking=True) for key, value in batch.items()
                if isinstance(value, torch.Tensor)
                and key not in ("disto_target", "token_to_center_atom", "r_set_to_rep_atom")
            }
            prediction = model(features, recycling_steps=recycling_steps)
            keys = [key for key in prediction if key.startswith("affinity_")]
            for key in keys:
                if prediction[key].numel() != len(batch["record"]) or not torch.isfinite(prediction[key]).all():
                    raise RuntimeError(f"Invalid BioIR output for {key} in batch {index}.")
            result = {key: prediction[key].float().cpu() for key in keys}
            result["exception"] = False
            writer.write_on_batch_end(None, None, result, None, batch, index, 0)
            completed += len(batch["record"])
        torch.cuda.synchronize()
        if completed != len(dataset):
            raise RuntimeError("BioIR scored record count differs from the valid manifest.")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        metadata = {
            "scoring_backend": "bioir", "bionemo_ir_version": version("bionemo-ir"),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "checkpoint_sha256": digest.hexdigest(),
            "seed": seed, "batch_size": batch_size, "recycling_steps": recycling_steps,
            "msa_subsample_rows": 1024, "fixed_pose": True, "affinity_heads": 2,
            "affinity_heads_dtype": "float32", "completed": completed,
            "failed_records": dataset.failed_records,
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        (output_dir.parent / "scoring_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

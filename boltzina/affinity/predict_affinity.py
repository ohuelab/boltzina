
from pathlib import Path
from dataclasses import asdict
from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, get_cache_path
from boltz.model.models.boltz2 import Boltz2
from pytorch_lightning import Trainer, seed_everything
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.data.types import Manifest

def predict_affinity(out_dir, output_dir = None, structures_dir = None, msa_dir = None, constraints_dir = None, template_dir = None, extra_mols_dir = None, manifest = None, affinity_checkpoint = None, sampling_steps_affinity=200, diffusion_samples_affinity=5, subsample_msa=True, num_subsampled_msa=1024, model="boltz2", step_scale=None, override=False, num_workers=1, strategy="auto", accelerator="gpu", devices=1, affinity_mw_correction=False, seed=None):
    out_dir = Path(out_dir)
    cache_dir = get_cache_path()
    cache_dir = Path(cache_dir)
    mol_dir = cache_dir/'mols'

    if seed is not None:
        seed_everything(seed)

    if affinity_checkpoint is None:
        affinity_checkpoint = cache_dir / "boltz2_aff.ckpt"

    if not affinity_checkpoint.exists():
        raise FileNotFoundError(f"Affinity checkpoint not found at {affinity_checkpoint}")


    predict_affinity_args = {
        "recycling_steps": 5,
        "sampling_steps": sampling_steps_affinity,
        "diffusion_samples": diffusion_samples_affinity,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    diffusion_params = Boltz2DiffusionParams()
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params.step_scale = step_scale
    pairformer_args = PairformerArgsV2()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )
    structures_dir = out_dir/"processed"/"structures" if structures_dir is None else structures_dir
    msa_dir = out_dir/"processed"/"msa" if msa_dir is None else msa_dir
    constraints_dir = out_dir/"processed"/"constraints" if constraints_dir is None else constraints_dir
    template_dir = out_dir/"processed"/"templates" if template_dir is None else template_dir
    extra_mols_dir = out_dir/"processed"/"mols" if extra_mols_dir is None else extra_mols_dir

    manifest = Manifest.load(out_dir / "processed" / "manifest.json" if manifest is None else manifest)

    output_dir = out_dir / "predictions" if output_dir is None else output_dir

    pred_writer = BoltzAffinityWriter(
        data_dir=structures_dir,
        output_dir=output_dir,
    )

    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=out_dir / "predictions",
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        num_workers=num_workers,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
        override_method="other",
        affinity=True,
    )

    model_module = Boltz2.load_from_checkpoint(
        affinity_checkpoint,
        strict=True,
        predict_args=predict_affinity_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args={"fk_steering": False, "guidance_update": False},
        affinity_mw_correction=affinity_mw_correction,
    )
    model_module.eval()


    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )
    return trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=True,
    )


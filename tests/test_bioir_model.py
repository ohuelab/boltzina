"""Fixed-coordinate and stochastic MSA contracts for the optional adapter."""

import importlib.util

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(importlib.util.find_spec("bionemo_ir") is None, reason="BioIR is optional")


def test_fixed_coordinates_preserve_compound_axis():
    from boltzina.affinity.bioir_model import fixed_coordinates

    coords = torch.arange(120.).reshape(2, 1, 20, 3)
    assert torch.equal(fixed_coordinates(coords), coords[:, 0])
    with pytest.raises(ValueError, match="one pose"):
        fixed_coordinates(torch.zeros(2, 2, 20, 3))


def test_msa_rows_are_resampled_per_call():
    from boltzina.affinity.bioir_model import SubsampledMSA

    class Capture(nn.Module):
        def forward(self, **kwargs):
            return kwargs["msa"]

    msa = torch.arange(72).reshape(2, 12, 3)
    torch.manual_seed(13)
    expected = [msa[:, torch.randperm(12)[:5]] for _ in range(2)]
    state = torch.get_rng_state()
    torch.manual_seed(13)
    module = SubsampledMSA(Capture(), count=5)
    for sample in expected:
        actual = module(torch.zeros(1), torch.zeros(1), msa, msa, msa, msa, msa, torch.ones(1))
        assert torch.equal(actual, sample)
    assert torch.equal(torch.get_rng_state(), state)


def test_fixed_pose_trunk_disables_automatic_cuda_graph_capture(monkeypatch):
    """CPU MSA sampling cannot be captured or frozen into a CUDA graph."""
    from boltzina.affinity.bioir_model import FixedPoseAffinity
    from bionemo_ir.models.boltz2 import Boltz2Affinity

    class ConfigChecked(Exception):
        pass

    def check_config(self, *, config, include_load_weights):
        assert config.trunk.graph_optimization_config is None
        assert config.skip_run_structure is True
        assert config.confidence_prediction is False
        raise ConfigChecked

    monkeypatch.setattr(Boltz2Affinity, '__init__', check_config)
    with pytest.raises(ConfigChecked):
        FixedPoseAffinity.from_checkpoint('unused.ckpt')


@pytest.mark.parametrize("cyclic", [False, True])
def test_relative_position_encoder_keeps_batch_axis(cyclic):
    from types import SimpleNamespace
    from boltzina.affinity.bioir_model import relative_position_features
    from bionemo_ir._torch.layers.position_encoders import RelativePositionEncoder

    class Features:
        def get_module_feed_dict(self, features, name):
            return dict(features)

    encoder = SimpleNamespace(
        period_broadcast=False, cyclic_pos_enc=True, r_max=32, s_max=2, fix_sym_check=False,
    )
    features = {
        "asym_id": torch.zeros(2, 6, dtype=torch.long),
        "entity_id": torch.zeros(2, 6, dtype=torch.long),
        "sym_id": torch.zeros(2, 6, dtype=torch.long),
        "residue_index": torch.arange(6).expand(2, -1),
        "token_index": torch.arange(6).expand(2, -1),
        "cyclic_period": torch.full((2, 6), 6. if cyclic else 0.),
    }
    actual = RelativePositionEncoder._relp_buckets(encoder, **relative_position_features(Features(), features))
    for row in range(2):
        single = {key: value[row:row+1] for key, value in features.items()}
        expected = RelativePositionEncoder._relp_buckets(encoder, **relative_position_features(Features(), single))
        for batched, separate in zip(actual, expected):
            torch.testing.assert_close(batched[row:row+1], separate)


@pytest.mark.gpu
def test_real_checkpoint_scores_two_fixed_poses(tmp_path):
    """Exercise the real loader, adapter and writer using bundled example poses."""
    import json
    from pathlib import Path
    from boltz.main import get_cache_path
    from boltzina.config import get_maxit_path, get_vina_path
    from boltzina.runner import BoltzinaRunner, RunnerConfig

    if not torch.cuda.is_available() or not (Path(get_cache_path()) / "boltz2_aff.ckpt").is_file():
        pytest.skip("A CUDA GPU and cached Boltz-2 affinity weights are required.")
    try:
        get_vina_path()
        get_maxit_path()
    except RuntimeError as error:
        pytest.skip(str(error))
    source = Path(__file__).parent.parent / "sample/CDK2/boltz_results_base"
    ligands = tmp_path / "ligands.smi"
    ligands.write_text("CCO ethanol\nc1ccccc1 benzene\n")
    out = tmp_path / "output"
    results = BoltzinaRunner(RunnerConfig(
        input_path=ligands, output_dir=out, work_dir=source,
        scoring_backend="bioir", seed=42, batch_size=2, num_workers=2,
        keep_intermediate_files=True,
    )).run()
    assert len(results) == 2
    assert results["affinity_probability_binary"].between(0, 1).all()
    metadata = json.loads((out / "boltz_out/scoring_metadata.json").read_text())
    assert metadata["completed"] == 2
    assert metadata["fixed_pose"] is True
    assert metadata["checkpoint_sha256"]

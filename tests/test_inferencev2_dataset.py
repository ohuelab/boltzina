from types import SimpleNamespace

import pytest

from boltzina.data.module import inferencev2


class FakeTokenizer:
    def tokenize(self, input_data):
        return SimpleNamespace(record_id=input_data.record_id)


class FakeCropper:
    def crop(self, tokenized, max_tokens, max_atoms):
        if tokenized.record_id == "bad":
            raise ValueError("no ligand tokens")
        return tokenized


def _fake_input(record, **kwargs):
    return SimpleNamespace(record_id=record.id, extra_mols={})


def test_affinity_dataset_prefilters_crop_failures(monkeypatch, tmp_path):
    records = [SimpleNamespace(id="bad"), SimpleNamespace(id="good")]
    manifest = SimpleNamespace(records=records)

    monkeypatch.setattr(inferencev2, "load_canonicals", lambda mol_dir: {})
    monkeypatch.setattr(inferencev2, "Boltz2Tokenizer", FakeTokenizer)
    monkeypatch.setattr(inferencev2, "AffinityCropper", FakeCropper)
    monkeypatch.setattr(inferencev2, "load_input", _fake_input)

    dataset = inferencev2.PredictionDataset(
        manifest=manifest,
        target_dir=tmp_path,
        msa_dir=tmp_path,
        mol_dir=tmp_path,
        affinity=True,
    )

    assert len(dataset) == 1
    assert dataset.valid_indices == [1]
    assert dataset.failed_records[0]["id"] == "bad"
    assert "Affinity cropper failed on bad" in dataset.failed_records[0]["error"]


def test_dataset_reports_record_id_without_recursive_fallback(monkeypatch, tmp_path):
    manifest = SimpleNamespace(records=[SimpleNamespace(id="rec0")])

    def fail_input(record, **kwargs):
        raise ValueError("missing pre_affinity")

    monkeypatch.setattr(inferencev2, "load_canonicals", lambda mol_dir: {})
    monkeypatch.setattr(inferencev2, "load_input", fail_input)

    dataset = inferencev2.PredictionDataset(
        manifest=manifest,
        target_dir=tmp_path,
        msa_dir=tmp_path,
        mol_dir=tmp_path,
        affinity=False,
    )

    with pytest.raises(RuntimeError, match="Input loading failed on rec0"):
        dataset[0]


def test_msa_cache_is_immutable_and_invalidates_changed_file(monkeypatch, tmp_path):
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class ParsedMSA:
        sequences: np.ndarray

    path = tmp_path / "msa.npz"
    path.write_bytes(b"one")
    calls = []

    def load(path):
        calls.append(path)
        return ParsedMSA(np.arange(len(path.read_bytes())))

    monkeypatch.setattr(inferencev2.MSA, "load", load)
    inferencev2._load_immutable_msa.cache_clear()
    try:
        first = inferencev2.load_cached_msa(path)
        assert inferencev2.load_cached_msa(path) is first
        with pytest.raises(ValueError, match="read-only"):
            first.sequences[0] = 9
        path.write_bytes(b"replacement")
        second = inferencev2.load_cached_msa(path)
        assert second is not first
        assert len(calls) == 2
    finally:
        inferencev2._load_immutable_msa.cache_clear()

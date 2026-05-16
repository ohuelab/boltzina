"""
Tests for boltzina.boltz_setup.generate_boltz_yaml — YAML generation.

Covers:
  - Single-chain: A + ligand B
  - Multi-chain: A, B, ... + ligand (next letter)
  - Correct YAML structure (version, sequences, properties)
  - Chain ID auto-assignment (sequential A, B, C...)
  - Affinity property targets the ligand chain
  - Sequence validation (invalid chars rejected)
  - YAML is valid and parseable
  - parse_yaml_ligand_chain() validation
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import types

import click
import pytest
import yaml

from boltzina.boltz_setup import (
    _prepare_yaml_for_boltz_run,
    extract_receptor_pdb,
    generate_boltz_yaml,
    parse_yaml_ligand_chain,
    run_boltz_predict,
)

CDK2_SEQUENCE = (
    "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIH"
    "TENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTTCDLK"
)
ROSCOVITINE_SMILES = "CCN(CC)c1nc(Nc2ccccc2)c2ncn(C(C)CO)c2n1"
SHORT_SEQ = CDK2_SEQUENCE[:50]


class TestGenerateBoltzYamlSingleChain:
    """Single protein chain: sequences=['SEQ'] → protein A, ligand B."""

    def test_creates_file(self, tmp_path):
        out = tmp_path / "test.yaml"
        yaml_path, ligand_chain_id = generate_boltz_yaml(
            sequences=[SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        assert yaml_path == out
        assert out.exists()

    def test_returns_ligand_chain_id_b(self, tmp_path):
        out = tmp_path / "test.yaml"
        _, ligand_chain_id = generate_boltz_yaml(
            sequences=[SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        assert ligand_chain_id == "B"

    def test_yaml_is_parseable(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_version_field(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["version"] == 1

    def test_protein_chain_id_is_a(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert len(protein_entries) == 1
        assert protein_entries[0]["protein"]["id"] == "A"

    def test_protein_sequence_present(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert protein_entries[0]["protein"]["sequence"] == SHORT_SEQ.upper()

    def test_default_yaml_marks_msa_empty(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert protein_entries[0]["protein"]["msa"] == "empty"

    def test_use_msa_server_leaves_msa_unspecified(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequences=[SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
            use_msa_server=True,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert "msa" not in protein_entries[0]["protein"]

    def test_ligand_chain_id_is_b(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        ligand_entries = [s for s in data["sequences"] if "ligand" in s]
        assert len(ligand_entries) == 1
        assert ligand_entries[0]["ligand"]["id"] == "B"

    def test_ligand_smiles_present(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        ligand_entries = [s for s in data["sequences"] if "ligand" in s]
        assert ligand_entries[0]["ligand"]["smiles"] == ROSCOVITINE_SMILES

    def test_affinity_property_targets_ligand(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        assert "properties" in data
        affinity_entries = [p for p in data["properties"] if "affinity" in p]
        assert len(affinity_entries) == 1
        assert affinity_entries[0]["affinity"]["binder"] == "B"

    def test_sequence_uppercased(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(sequences=[SHORT_SEQ.lower()], representative_smiles=ROSCOVITINE_SMILES, output_path=out)
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert protein_entries[0]["protein"]["sequence"] == SHORT_SEQ.upper()

    def test_invalid_sequence_raises(self, tmp_path):
        out = tmp_path / "test.yaml"
        with pytest.raises(ValueError, match="non-standard"):
            generate_boltz_yaml(
                sequences=["MENFQKV123INVALID"],
                representative_smiles=ROSCOVITINE_SMILES,
                output_path=out,
            )

    def test_empty_sequences_raises(self, tmp_path):
        out = tmp_path / "test.yaml"
        with pytest.raises(ValueError, match="At least one protein sequence"):
            generate_boltz_yaml(sequences=[], representative_smiles=ROSCOVITINE_SMILES, output_path=out)


class TestGenerateBoltzYamlMultiChain:
    """Multi-chain: sequences=['SEQ1','SEQ2'] → proteins A, B; ligand C."""

    def test_two_chains_returns_ligand_c(self, tmp_path):
        out = tmp_path / "test.yaml"
        _, ligand_chain_id = generate_boltz_yaml(
            sequences=[SHORT_SEQ, SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        assert ligand_chain_id == "C"

    def test_three_chains_returns_ligand_d(self, tmp_path):
        out = tmp_path / "test.yaml"
        _, ligand_chain_id = generate_boltz_yaml(
            sequences=[SHORT_SEQ, SHORT_SEQ, SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        assert ligand_chain_id == "D"

    def test_two_chains_have_sequential_ids(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequences=[SHORT_SEQ, SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        ids = [e["protein"]["id"] for e in protein_entries]
        assert ids == ["A", "B"]

    def test_two_chains_affinity_binder_is_c(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequences=[SHORT_SEQ, SHORT_SEQ],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        affinity_entries = [p for p in data["properties"] if "affinity" in p]
        assert affinity_entries[0]["affinity"]["binder"] == "C"

    def test_two_chains_have_correct_sequences(self, tmp_path):
        seq_a = "MENFQK"
        seq_b = "AKLSILP"
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequences=[seq_a, seq_b],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert protein_entries[0]["protein"]["sequence"] == seq_a.upper()
        assert protein_entries[1]["protein"]["sequence"] == seq_b.upper()


class TestParseYamlLigandChain:
    """Tests for parse_yaml_ligand_chain()."""

    def _write_yaml(self, tmp_path, content: str) -> Path:
        p = tmp_path / "input.yaml"
        p.write_text(content)
        return p

    def test_extracts_smiles_and_chain(self, tmp_path):
        p = self._write_yaml(tmp_path, """
version: 1
sequences:
  - protein:
      id: A
      sequence: MENFQKV
  - ligand:
      id: B
      smiles: 'CCO'
properties:
  - affinity:
      binder: B
""")
        smiles, chain_id = parse_yaml_ligand_chain(p)
        assert smiles == "CCO"
        assert chain_id == "B"

    def test_missing_affinity_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, """
version: 1
sequences:
  - protein:
      id: A
      sequence: MENFQKV
  - ligand:
      id: B
      smiles: 'CCO'
""")
        with pytest.raises(ValueError, match="properties.affinity.binder"):
            parse_yaml_ligand_chain(p)

    def test_missing_protein_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, """
version: 1
sequences:
  - ligand:
      id: B
      smiles: 'CCO'
properties:
  - affinity:
      binder: B
""")
        with pytest.raises(ValueError, match="protein"):
            parse_yaml_ligand_chain(p)

    def test_missing_ligand_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, """
version: 1
sequences:
  - protein:
      id: A
      sequence: MENFQKV
properties:
  - affinity:
      binder: B
""")
        with pytest.raises(ValueError, match="binder"):
            parse_yaml_ligand_chain(p)

    def test_sample_yaml(self):
        """The existing sample YAML should parse without errors."""
        from tests.conftest import SAMPLE_CDK2
        yaml_path = SAMPLE_CDK2 / "1ckp_cdk2.yaml"
        if yaml_path.exists():
            smiles, chain_id = parse_yaml_ligand_chain(yaml_path)
            assert chain_id is not None
            assert smiles is not None


class TestRunBoltzPredict:
    """Tests for the Boltz predict wrapper."""

    def test_calls_click_callback_when_predict_is_click_command(self, tmp_path, monkeypatch):
        calls = {}

        def fake_predict_callback(**kwargs):
            calls.update(kwargs)
            work_dir = Path(kwargs["out_dir"]) / "boltz_results_input"
            work_dir.mkdir(parents=True)

        fake_boltz = types.ModuleType("boltz")
        fake_boltz_main = types.ModuleType("boltz.main")
        fake_boltz_main.predict = click.Command(
            "predict",
            callback=fake_predict_callback,
        )
        monkeypatch.setitem(sys.modules, "boltz", fake_boltz)
        monkeypatch.setitem(sys.modules, "boltz.main", fake_boltz_main)

        yaml_path = tmp_path / "input.yaml"
        yaml_path.write_text("version: 1\n")
        out_dir = tmp_path / "out"

        work_dir = run_boltz_predict(
            yaml_path=yaml_path,
            out_dir=out_dir,
            cache=tmp_path / "cache",
            seed=123,
            no_kernels=True,
        )

        from rdkit.Chem import AllChem

        assert work_dir == out_dir / "boltz_results_input"
        assert calls["data"] == str(yaml_path)
        assert calls["out_dir"] == str(out_dir)
        assert calls["cache"] == str(tmp_path / "cache")
        assert calls["seed"] == 123
        assert calls["no_kernels"] is True
        assert calls["model"] == "boltz2"
        assert hasattr(AllChem, "Descriptors")


class TestPrepareYamlForBoltzRun:
    """Tests for YAML preparation before invoking Boltz."""

    def test_adds_empty_msa_when_server_disabled(self, tmp_path):
        src = tmp_path / "input.yaml"
        dest = tmp_path / "work" / "input.yaml"
        src.write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            "      sequence: MENFQKV\n"
            "  - ligand:\n"
            "      id: B\n"
            "      smiles: CCO\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n"
        )

        _prepare_yaml_for_boltz_run(src, dest, use_msa_server=False)

        with open(dest) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert protein_entries[0]["protein"]["msa"] == "empty"

    def test_preserves_missing_msa_when_server_enabled(self, tmp_path):
        src = tmp_path / "input.yaml"
        dest = tmp_path / "work" / "input.yaml"
        src.write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            "      sequence: MENFQKV\n"
            "  - ligand:\n"
            "      id: B\n"
            "      smiles: CCO\n"
        )

        _prepare_yaml_for_boltz_run(src, dest, use_msa_server=True)

        with open(dest) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert "msa" not in protein_entries[0]["protein"]


class TestExtractReceptorPdb:
    """Tests for predicted receptor extraction."""

    def test_converts_cif_when_preexisting_protein_pdb_absent(self, tmp_path, sample_cdk2_dir):
        fname = "1ckp_cdk2"
        pred_dir = tmp_path / "predictions" / fname
        pred_dir.mkdir(parents=True)
        source_cif = sample_cdk2_dir / "boltz_results_base" / "predictions" / fname / f"{fname}_model_0.cif"
        shutil.copy2(source_cif, pred_dir / f"{fname}_model_0.cif")

        receptor_pdb = extract_receptor_pdb(tmp_path, fname)

        assert receptor_pdb == pred_dir / f"{fname}_model_0_protein.pdb"
        text = receptor_pdb.read_text()
        assert text.startswith("ATOM")
        assert "HETATM" not in text

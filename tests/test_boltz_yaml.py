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

import pytest
import yaml

from boltzina.boltz_setup import generate_boltz_yaml, parse_yaml_ligand_chain

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

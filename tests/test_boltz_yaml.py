"""
Tests for boltzina.boltz_setup.generate_boltz_yaml — YAML generation.

Covers:
  - Correct YAML structure (version, sequences, properties)
  - Protein chain ID and sequence
  - Ligand chain ID and SMILES
  - Affinity property targeting ligand chain
  - Sequence validation (invalid chars rejected)
  - YAML is valid and parseable
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from boltzina.boltz_setup import generate_boltz_yaml

CDK2_SEQUENCE = (
    "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTTCDLKICDFGLARVADPDHDHTGFLTEYVATRWYRAPEVLLGSRHYSTGVDIWSVGCIFAEMCNRKPIFKGSDYLDQLNRFVTLGTP"
)
ROSCOVITINE_SMILES = "CCN(CC)c1nc(Nc2ccccc2)c2ncn(C(C)CO)c2n1"


class TestGenerateBoltzYaml:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "test.yaml"
        result = generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        assert result == out
        assert out.exists()

    def test_yaml_is_parseable(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_version_field(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["version"] == 1

    def test_protein_chain_in_sequences(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
            protein_chain_id="A",
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        assert len(protein_entries) == 1
        prot = protein_entries[0]["protein"]
        assert prot["sequence"] == CDK2_SEQUENCE[:50].upper()

    def test_ligand_chain_in_sequences(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
            ligand_chain_id="B",
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        ligand_entries = [s for s in data["sequences"] if "ligand" in s]
        assert len(ligand_entries) == 1
        lig = ligand_entries[0]["ligand"]
        assert lig["smiles"] == ROSCOVITINE_SMILES

    def test_affinity_property_targets_ligand(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
            ligand_chain_id="B",
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert "properties" in data
        affinity_entries = [p for p in data["properties"] if "affinity" in p]
        assert len(affinity_entries) == 1
        assert affinity_entries[0]["affinity"]["binder"] == "B"

    def test_custom_chain_ids(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50],
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
            protein_chain_id="C",
            ligand_chain_id="D",
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        affinity_entries = [p for p in data["properties"] if "affinity" in p]
        assert affinity_entries[0]["affinity"]["binder"] == "D"

    def test_invalid_sequence_raises(self, tmp_path):
        out = tmp_path / "test.yaml"
        with pytest.raises(ValueError, match="non-standard"):
            generate_boltz_yaml(
                sequence="MENFQKV123INVALID",
                representative_smiles=ROSCOVITINE_SMILES,
                output_path=out,
            )

    def test_sequence_uppercased(self, tmp_path):
        out = tmp_path / "test.yaml"
        generate_boltz_yaml(
            sequence=CDK2_SEQUENCE[:50].lower(),
            representative_smiles=ROSCOVITINE_SMILES,
            output_path=out,
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        protein_entries = [s for s in data["sequences"] if "protein" in s]
        prot = protein_entries[0]["protein"]
        assert prot["sequence"] == CDK2_SEQUENCE[:50].upper()

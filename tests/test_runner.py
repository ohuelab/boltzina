"""
Tests for boltzina.runner — BoltzinaRunner orchestration logic.

Covers:
  - _resolve_sequences(): single, colon-split, FASTA single/multi
  - _get_reference_smiles(): from string, from SDF, default (None)
  - _prepare_ligands(): raises ValueError on empty result
  - _parse_fasta(): single and multi-entry FASTA
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from boltzina.runner import BoltzinaRunner, RunnerConfig, _parse_fasta


# ---------------------------------------------------------------------------
# _parse_fasta
# ---------------------------------------------------------------------------

class TestParseFasta:
    def test_single_entry(self, tmp_path):
        p = tmp_path / "single.fasta"
        p.write_text(">entry1\nMENFQKV\n")
        seqs = _parse_fasta(p)
        assert seqs == ["MENFQKV"]

    def test_multi_entry(self, tmp_path):
        p = tmp_path / "multi.fasta"
        p.write_text(">A\nMENFQKV\n>B\nAKLSILP\n")
        seqs = _parse_fasta(p)
        assert seqs == ["MENFQKV", "AKLSILP"]

    def test_multiline_sequence(self, tmp_path):
        p = tmp_path / "multi_line.fasta"
        p.write_text(">A\nMENFQ\nKVEKI\n")
        seqs = _parse_fasta(p)
        assert seqs == ["MENFQKVEKI"]

    def test_uppercased(self, tmp_path):
        p = tmp_path / "lower.fasta"
        p.write_text(">A\nmenfqkv\n")
        seqs = _parse_fasta(p)
        assert seqs == ["MENFQKV"]

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.fasta"
        p.write_text("")
        with pytest.raises(ValueError, match="No sequences"):
            _parse_fasta(p)

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "blanks.fasta"
        p.write_text("\n>A\n\nMENFQKV\n\n")
        seqs = _parse_fasta(p)
        assert seqs == ["MENFQKV"]


# ---------------------------------------------------------------------------
# _resolve_sequences
# ---------------------------------------------------------------------------

class TestResolveSequences:
    def _make_runner(self, **kwargs):
        cfg = RunnerConfig(
            input_path=Path("/dev/null"),
            output_dir=Path("/dev/null"),
            **kwargs,
        )
        return BoltzinaRunner(cfg)

    def test_single_sequence(self):
        runner = self._make_runner(sequence="MENFQKV")
        seqs = runner._resolve_sequences()
        assert seqs == ["MENFQKV"]

    def test_colon_split_two_chains(self):
        runner = self._make_runner(sequence="MENFQKV:AKLSILP")
        seqs = runner._resolve_sequences()
        assert seqs == ["MENFQKV", "AKLSILP"]

    def test_colon_split_three_chains(self):
        runner = self._make_runner(sequence="SEQ1:SEQ2:SEQ3")
        seqs = runner._resolve_sequences()
        assert seqs == ["SEQ1", "SEQ2", "SEQ3"]

    def test_fasta_single_chain(self, fasta_file):
        runner = self._make_runner(sequence_file=fasta_file)
        seqs = runner._resolve_sequences()
        assert len(seqs) == 1
        assert seqs[0].startswith("MENFQKV")

    def test_fasta_multichain(self, multichain_fasta_file):
        runner = self._make_runner(sequence_file=multichain_fasta_file)
        seqs = runner._resolve_sequences()
        assert len(seqs) == 2

    def test_sequence_stripped(self):
        runner = self._make_runner(sequence="  MENFQKV  ")
        seqs = runner._resolve_sequences()
        assert seqs == ["MENFQKV"]


# ---------------------------------------------------------------------------
# _get_reference_smiles
# ---------------------------------------------------------------------------

class TestGetReferenceSmiles:
    def _make_runner(self, reference_ligand=None, input_path=None):
        cfg = RunnerConfig(
            input_path=input_path or Path("/dev/null"),
            output_dir=Path("/dev/null"),
            sequence="MENFQKV",
            reference_ligand=reference_ligand,
        )
        return BoltzinaRunner(cfg)

    def test_returns_none_when_not_set(self):
        runner = self._make_runner()
        assert runner._get_reference_smiles() is None

    def test_smiles_string_returned_as_is(self):
        runner = self._make_runner(reference_ligand="CCO")
        assert runner._get_reference_smiles() == "CCO"

    def test_nonexistent_path_treated_as_smiles(self):
        """A path that doesn't exist is treated as a SMILES string, not a file."""
        runner = self._make_runner(reference_ligand="c1ccccc1")
        result = runner._get_reference_smiles()
        assert result == "c1ccccc1"

    def test_sdf_file_reads_smiles(self, simple_sdf_file):
        runner = self._make_runner(reference_ligand=str(simple_sdf_file))
        smiles = runner._get_reference_smiles()
        assert smiles is not None
        assert len(smiles) > 0


# ---------------------------------------------------------------------------
# _prepare_ligands — empty check
# ---------------------------------------------------------------------------

class TestPrepareLigandsEmptyCheck:
    def test_empty_ligands_raises(self, tmp_path):
        """_prepare_ligands() must raise ValueError when no ligands are prepared."""
        empty_smi = tmp_path / "empty.smi"
        empty_smi.write_text("# comment only\n")

        cfg = RunnerConfig(
            input_path=empty_smi,
            output_dir=tmp_path / "out",
            work_dir=Path("/dev/null"),  # avoid running predict
        )
        runner = BoltzinaRunner(cfg)
        runner._work_dir = Path("/dev/null")
        runner._receptor_pdb = Path("/dev/null")
        runner._fname = "test"
        runner._ligand_chain_id = "B"

        with pytest.raises(ValueError, match="No ligands"):
            runner._prepare_ligands()

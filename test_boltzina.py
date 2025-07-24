import unittest
import tempfile
import shutil
import os
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd

from boltzina_main import Boltzina


class TestBoltzina(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.receptor_pdbqt = Path(self.temp_dir) / "receptor.pdbqt"
        self.output_dir = Path(self.temp_dir) / "output"

        # Create mock receptor file
        self.receptor_pdbqt.write_text("MOCK RECEPTOR PDBQT CONTENT")

        # Create mock CCD
        self.mock_ccd = {"TEST": Mock()}

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('boltzina_main.get_cache_path')
    def test_init(self, mock_get_cache_path):
        mock_cache_dir = Path(self.temp_dir) / "cache"
        mock_cache_dir.mkdir()
        mock_get_cache_path.return_value = mock_cache_dir

        # Create mock CCD file
        ccd_path = mock_cache_dir / "ccd.pkl"
        with open(ccd_path, 'wb') as f:
            pickle.dump(self.mock_ccd, f)

        boltzina = Boltzina(
            receptor_pdbqt=str(self.receptor_pdbqt),
            output_dir=str(self.output_dir),
            exhaustiveness=4
        )

        self.assertEqual(boltzina.receptor_pdbqt, self.receptor_pdbqt)
        self.assertEqual(boltzina.output_dir, self.output_dir)
        self.assertEqual(boltzina.exhaustiveness, 4)
        self.assertTrue(self.output_dir.exists())

    @patch('boltzina_main.get_cache_path')
    def test_load_ccd_existing(self, mock_get_cache_path):
        mock_cache_dir = Path(self.temp_dir) / "cache"
        mock_cache_dir.mkdir()
        mock_get_cache_path.return_value = mock_cache_dir

        ccd_path = mock_cache_dir / "ccd.pkl"
        with open(ccd_path, 'wb') as f:
            pickle.dump(self.mock_ccd, f)

        boltzina = Boltzina(
            receptor_pdbqt=str(self.receptor_pdbqt),
            output_dir=str(self.output_dir)
        )

        self.assertEqual(boltzina.ccd, self.mock_ccd)

    @patch('boltzina_main.get_cache_path')
    def test_load_ccd_not_existing(self, mock_get_cache_path):
        mock_cache_dir = Path(self.temp_dir) / "cache"
        mock_cache_dir.mkdir()
        mock_get_cache_path.return_value = mock_cache_dir

        boltzina = Boltzina(
            receptor_pdbqt=str(self.receptor_pdbqt),
            output_dir=str(self.output_dir)
        )

        self.assertEqual(boltzina.ccd, {})

    @patch('subprocess.run')
    def test_convert_to_pdbqt_sdf(self, mock_run):
        boltzina = Boltzina.__new__(Boltzina)

        input_file = Path("test.sdf")
        output_file = Path("test.pdbqt")

        boltzina._convert_to_pdbqt(input_file, output_file, "sdf")

        mock_run.assert_called_once_with(
            ["obabel", "test.sdf", "-O", "test.pdbqt"],
            check=True
        )

    def test_convert_to_pdbqt_unsupported(self):
        boltzina = Boltzina.__new__(Boltzina)

        input_file = Path("test.xyz")
        output_file = Path("test.pdbqt")

        with self.assertRaises(ValueError):
            boltzina._convert_to_pdbqt(input_file, output_file, "xyz")

    @patch('subprocess.run')
    def test_run_vina(self, mock_run):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.receptor_pdbqt = Path("receptor.pdbqt")
        boltzina.exhaustiveness = 8

        ligand_pdbqt = Path("ligand.pdbqt")
        output_pdbqt = Path("output.pdbqt")

        boltzina._run_vina(ligand_pdbqt, output_pdbqt)

        expected_cmd = [
            "vina",
            "--receptor", "receptor.pdbqt",
            "--ligand", "ligand.pdbqt",
            "--out", "output.pdbqt",
            "--exhaustiveness", "8"
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)

    def test_extract_docking_score(self):
        boltzina = Boltzina.__new__(Boltzina)

        # Create mock docked PDBQT file
        pdbqt_content = """MODEL 1
REMARK VINA RESULT:    -8.5      0.000      0.000
ATOM      1  C   MOL A   1      20.000  20.000  20.000  1.00 20.00           C
ENDMDL
MODEL 2
REMARK VINA RESULT:    -7.2      1.500      2.100
ATOM      1  C   MOL A   1      21.000  21.000  21.000  1.00 20.00           C
ENDMDL
"""

        pdbqt_file = Path(self.temp_dir) / "test.pdbqt"
        pdbqt_file.write_text(pdbqt_content)

        # Test extracting score for pose 0
        score = boltzina._extract_docking_score(pdbqt_file, 0)
        self.assertEqual(score, -8.5)

        # Test extracting score for pose 1
        score = boltzina._extract_docking_score(pdbqt_file, 1)
        self.assertEqual(score, -7.2)

        # Test extracting score for non-existent pose
        score = boltzina._extract_docking_score(pdbqt_file, 5)
        self.assertIsNone(score)

    @patch('boltzina_main.Chem.MolFromPDBFile')
    def test_update_ccd_for_ligand(self, mock_mol_from_pdb):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.ccd = {}

        # Create test directory structure
        ligand_output_dir = Path(self.temp_dir) / "0"
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        docked_ligands_dir.mkdir(parents=True)

        # Create mock PDB file
        pdb_file = docked_ligands_dir / "docked_ligand_0.pdb"
        pdb_file.write_text("MOCK PDB")

        # Mock RDKit molecule
        mock_mol = Mock()
        mock_atom = Mock()
        mock_pdb_info = Mock()
        mock_pdb_info.GetName.return_value = " C1 "
        mock_atom.GetPDBResidueInfo.return_value = mock_pdb_info
        mock_mol.GetAtoms.return_value = [mock_atom]
        mock_mol_from_pdb.return_value = mock_mol

        boltzina._update_ccd_for_ligand(0, ligand_output_dir)

        self.assertEqual(boltzina.ccd["MOL"], mock_mol)
        mock_atom.SetProp.assert_called_once_with("name", "C1")

    def test_save_results_csv(self):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.output_dir = Path(self.temp_dir)
        boltzina.results = [
            {
                'ligand_name': 'test_ligand',
                'ligand_idx': 0,
                'docking_rank': 1,
                'docking_score': -8.5,
                'affinity_pred_value': -0.144,
                'affinity_probability_binary': 0.216,
                'affinity_pred_value1': 0.004,
                'affinity_probability_binary1': 0.059,
                'affinity_pred_value2': -0.292,
                'affinity_probability_binary2': 0.373
            }
        ]

        boltzina.save_results_csv()

        csv_file = Path(self.temp_dir) / "boltzina_results.csv"
        self.assertTrue(csv_file.exists())

        # Read and verify CSV content
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['ligand_name'], 'test_ligand')
        self.assertEqual(df.iloc[0]['docking_score'], -8.5)

    def test_save_results_csv_no_results(self):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.output_dir = Path(self.temp_dir)
        boltzina.results = []

        with patch('builtins.print') as mock_print:
            boltzina.save_results_csv()
            mock_print.assert_called_with("No results to save")

    def test_get_results_dataframe(self):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.results = [
            {'ligand_name': 'test1', 'docking_score': -8.5},
            {'ligand_name': 'test2', 'docking_score': -7.2}
        ]

        df = boltzina.get_results_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['ligand_name'], 'test1')
        self.assertEqual(df.iloc[1]['docking_score'], -7.2)

    @patch('subprocess.run')
    def test_process_pose(self, mock_run):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.receptor_pdbqt = Path("receptor.pdbqt")

        # Create test directory structure
        ligand_output_dir = Path(self.temp_dir) / "0"
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        docked_ligands_dir.mkdir(parents=True)

        pdb_file = docked_ligands_dir / "docked_ligand_0.pdb"
        pdb_file.write_text("MOCK PDB")

        boltzina._process_pose(0, "0", pdb_file)

        # Verify that subprocess.run was called with expected commands
        self.assertEqual(mock_run.call_count, 4)  # 4 commands should be run

    @patch('boltzina_main.calc_from_data')
    @patch('builtins.open')
    def test_score_poses(self, mock_open, mock_calc_from_data):
        boltzina = Boltzina.__new__(Boltzina)
        boltzina.ccd = self.mock_ccd
        boltzina.results = []

        # Create test directory structure
        ligand_output_dir = Path(self.temp_dir) / "0"
        docked_ligands_dir = ligand_output_dir / "docked_ligands"
        boltz_output_dir = ligand_output_dir / "boltz_out"
        docked_ligands_dir.mkdir(parents=True)
        boltz_output_dir.mkdir(parents=True)

        # Create mock complex file
        complex_file = docked_ligands_dir / "docked_ligand_0_B_complex_fix.cif"
        complex_file.write_text("MOCK CIF")

        # Create mock affinity results directory and file
        affinity_dir = boltz_output_dir / "0_0"
        affinity_dir.mkdir()
        affinity_file = affinity_dir / "affinity_0_0.json"
        affinity_file.write_text('{"affinity_pred_value": -0.144}')

        # Mock calc_from_data return
        mock_calc_from_data.return_value = {"success": True}

        # Mock docking score extraction
        with patch.object(boltzina, '_extract_docking_score', return_value=-8.5):
            boltzina._score_poses(0, ligand_output_dir, "test_ligand")

        # Verify calc_from_data was called
        mock_calc_from_data.assert_called_once()

        # Verify results were added
        self.assertEqual(len(boltzina.results), 1)
        self.assertEqual(boltzina.results[0]['ligand_name'], "test_ligand")
        self.assertEqual(boltzina.results[0]['affinity_pred_value'], -0.144)


class TestBoltzinaIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.receptor_pdbqt = Path(self.temp_dir) / "receptor.pdbqt"
        self.ligand_sdf = Path(self.temp_dir) / "ligand.sdf"
        self.output_dir = Path(self.temp_dir) / "output"

        # Create mock files
        self.receptor_pdbqt.write_text("MOCK RECEPTOR")
        self.ligand_sdf.write_text("MOCK SDF")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('boltzina_main.get_cache_path')
    @patch('subprocess.run')
    @patch('boltzina_main.calc_from_data')
    @patch('boltzina_main.Chem.MolFromPDBFile')
    def test_full_pipeline_mock(self, mock_mol_from_pdb, mock_calc_from_data,
                               mock_subprocess, mock_get_cache_path):
        # Setup mocks
        mock_cache_dir = Path(self.temp_dir) / "cache"
        mock_cache_dir.mkdir()
        mock_get_cache_path.return_value = mock_cache_dir

        ccd_path = mock_cache_dir / "ccd.pkl"
        with open(ccd_path, 'wb') as f:
            pickle.dump({}, f)

        # Mock RDKit molecule
        mock_mol = Mock()
        mock_atom = Mock()
        mock_pdb_info = Mock()
        mock_pdb_info.GetName.return_value = " C1 "
        mock_atom.GetPDBResidueInfo.return_value = mock_pdb_info
        mock_mol.GetAtoms.return_value = [mock_atom]
        mock_mol_from_pdb.return_value = mock_mol

        # Mock calc_from_data
        mock_calc_from_data.return_value = {"success": True}

        # Create mock docked PDBQT content for scoring
        def mock_subprocess_side_effect(*args, **kwargs):
            if args[0][0] == "obabel" and "-m" in args[0]:
                # Create mock docked ligand PDB files
                output_pattern = args[0][-1]  # Last argument is output pattern
                output_dir = Path(output_pattern).parent
                (output_dir / "docked_ligand_0.pdb").write_text("MOCK DOCKED PDB")
            elif args[0][0] == "vina":
                # Create mock docked PDBQT file
                output_file = Path(args[0][args[0].index("--out") + 1])
                docked_content = """MODEL 1
REMARK VINA RESULT:    -8.5      0.000      0.000
ATOM      1  C   MOL A   1      20.000  20.000  20.000  1.00 20.00           C
ENDMDL
"""
                output_file.write_text(docked_content)

        mock_subprocess.side_effect = mock_subprocess_side_effect

        # Mock affinity results file creation
        def mock_calc_side_effect(cif_path, output_dir, fname, ccd, work_dir):
            result_dir = Path(output_dir) / fname
            result_dir.mkdir(parents=True, exist_ok=True)
            affinity_file = result_dir / f"affinity_{fname}.json"
            affinity_data = {
                "affinity_pred_value": -0.144,
                "affinity_probability_binary": 0.216
            }
            with open(affinity_file, 'w') as f:
                json.dump(affinity_data, f)
            return {"success": True}

        mock_calc_from_data.side_effect = mock_calc_side_effect

        # Run the pipeline
        boltzina = Boltzina(
            receptor_pdbqt=str(self.receptor_pdbqt),
            output_dir=str(self.output_dir),
            exhaustiveness=2
        )

        boltzina.run([str(self.ligand_sdf)], "sdf")

        # Verify results
        self.assertEqual(len(boltzina.results), 1)
        self.assertEqual(boltzina.results[0]['ligand_name'], "ligand")
        self.assertEqual(boltzina.results[0]['docking_score'], -8.5)
        self.assertEqual(boltzina.results[0]['affinity_pred_value'], -0.144)

        # Test CSV output
        boltzina.save_results_csv()
        csv_file = self.output_dir / "boltzina_results.csv"
        self.assertTrue(csv_file.exists())


if __name__ == '__main__':
    unittest.main()

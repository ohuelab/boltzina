import argparse
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from boltz.data.parse.schema import compute_3d_conformer
from pathlib import Path
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

def prepare_mol(smiles, output_path):
    try:
        mol_2d = Chem.MolFromSmiles(smiles)
        if mol_2d is None:
            print(f"Failed to parse smiles: {smiles}")
            return None
        mol_add_h = Chem.AddHs(mol_2d)
        success = compute_3d_conformer(mol_add_h)
        if not success:
            print(f"Failed to compute 3D conformer for {mol_2d}")
        mol_3d = Chem.RemoveHs(mol_add_h)

        # Assign canonical atom ordering for consistent naming
        canonical_order = AllChem.CanonicalRankAtoms(mol_3d)
        Chem.AssignStereochemistry(mol_3d, force=True, cleanIt=True)
        pdb_resn, pdb_chain, pdb_resi = "UNL", "A", 1
        for atom, can_idx in zip(mol_3d.GetAtoms(), canonical_order):
            # Generate atom names using element symbol + canonical index
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                msg = (
                    f"{seq} has an atom with a name longer than "
                    f"4 characters: {atom_name}."
                )
                raise ValueError(msg)
            atom.SetProp("name", atom_name)
            info = atom.GetPDBResidueInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()

            # Set PDB atom information (name, residue, chain) for proper file format
            info.SetName(atom_name.rjust(4))  # Right-justify to 4 characters for PDB format
            info.SetResidueName(pdb_resn)
            info.SetResidueNumber(pdb_resi)
            info.SetChainId(pdb_chain)
            info.SetIsHeteroAtom(True)

            atom.SetMonomerInfo(info)

        Chem.MolToPDBFile(mol_3d, output_path)
        return mol_3d
    except Exception as e:
        print(f"Failed to prepare mol {output_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_smi_path", type=str, help="Input SMILES file path")
    parser.add_argument("--ligand_prefix", type=str, default=None, help="Ligand prefix")
    parser.add_argument("--output_dir", type=str, default="out", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ligand_prefix = args.ligand_prefix
    input_smi_path = Path(args.input_smi_path)

    input_smiles_dict = {}
    with open(input_smi_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            smiles = parts[0]
            if len(parts) < 2 and ligand_prefix is None:
                raise ValueError("Ligand prefix is required if name isn't provided")
            name = parts[1] if len(parts) >= 2 else f"{ligand_prefix}{i}"
            input_smiles_dict[name] = smiles

    (output_dir/"input_pdbs").mkdir(parents=True, exist_ok=True)

    mols_dict = {}
    for name, smiles in input_smiles_dict.items():
        output_path = output_dir/ "input_pdbs" / f"{name}.pdb"
        mol_prepared = prepare_mol(smiles, output_path)
        if mol_prepared is None:
            continue
        mols_dict[name] = mol_prepared

    with open(output_dir/ "prepared_mols.pkl", "wb") as f:
        pickle.dump(mols_dict, f)
    print(f"Prepared {len(mols_dict)}/{len(input_smiles_dict)} mols")

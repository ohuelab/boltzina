import pickle
from pathlib import Path
from rdkit.Chem import AllChem

from boltz.data.parse.mmcif import parse_mmcif
from boltz.main import get_cache_path
from boltz.data.parse.schema import compute_3d_conformer
from boltzina.affinity.predict_affinity import predict_affinity

def preprocess_mol(smiles_seq):
    mol = AllChem.MolFromSmiles(smiles_seq)
    mol = AllChem.AddHs(mol)
    canonical_order = AllChem.CanonicalRankAtoms(mol)

    canonical_order = AllChem.CanonicalRankAtoms(mol)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            msg = (
                f"{smiles_seq} has an atom with a name longer than "
                f"4 characters: {atom_name}."
            )
            raise ValueError(msg)
        atom.SetProp("name", atom_name)
    success = compute_3d_conformer(mol)
    if not success:
        msg = f"Failed to compute 3D conformer for {smiles_seq}"
        raise ValueError(msg)

    mol_no_h = AllChem.RemoveHs(mol, sanitize=False)
    return mol_no_h

def calc_from_data(cif_path, fname, ccd, work_dir, output_dir, seed=None):
    work_dir = Path(work_dir)
    output_dir = Path(output_dir)
    cache_dir = get_cache_path()
    moldir = cache_dir / 'mols'
    cif_path = Path(cif_path)
    parsed_structure = parse_mmcif(
        path=cif_path,
        mols=ccd,
        moldir=moldir,
        call_compute_interfaces=False
    )
    structure_v2 = parsed_structure.data
    output_path = output_dir / f"{fname}.npz"
    structure_v2.dump(output_path)
    prediction = predict_affinity(work_dir, output_dir = output_dir, structures_dir = output_dir, seed=seed)
    return prediction

if __name__ == "__main__":
    fname = "oscoi1a_osjazp1_cfa_data"
    mol_file = None
    ligand_name = "LIG1"
    smiles_seq = "C1CCCCC1"
    cache_dir = get_cache_path()
    ccd_path = cache_dir / 'ccd.pkl'

    with ccd_path.open('rb') as file:
        ccd = pickle.load(file)

    if mol_file is not None:
        with mol_file.open('rb') as f:
            mol = pickle.load(f)
        for k in mol:
            ccd[k] = mol[k]
    else:
        ccd[ligand_name] = preprocess_mol(smiles_seq)

    calc_from_data(
        cif_path="bolts_results/boltz_results_oscoi1a_osjazp1_cfa_data/predictions/oscoi1a_osjazp1_cfa_data/oscoi1a_osjazp1_cfa_data_model_0.cif",
        fname=fname,
        ccd=ccd,
        work_dir="bolts_results/boltz_results_oscoi1a_osjazp1_cfa_data/predictions/oscoi1a_osjazp1_cfa_data"
    )

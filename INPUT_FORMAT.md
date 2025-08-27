# Input Format Requirements

There are several input requirements for actual predictions. We plan to provide detailed procedures for this in the future.

There are files in `sample/CDK2` that follow the actual format.

The `receptor_pdb` must strictly match the atom count and sequence count from Boltz-2. For example, we recommend directly using PDB files converted from CIF files obtained from Boltz-2.

Currently, ligands support PDB format. Here, it is important that each atom has a unique atom name assigned (such as `N1`, `C1`, etc. below) to match the Boltz-2 input format.

```pdb
COMPND    BDB50293153
HETATM    1  N1  UNL     1       2.365  -1.118  -0.787  1.00  0.00           N
HETATM    2  C1  UNL     1       3.062   0.001  -0.984  1.00  0.00           C
HETATM    3  C2  UNL     1       2.392   1.154  -0.524  1.00  0.00           C
...
```

If errors occur with PDB only, you can also pass mol_dict (a dictionary of ligand names and RDKit Mol objects). In this case, assign a `"name"` property to each atom to accurately correspond to the PDB above.

```python
if self.mol_dict:
    mol = self.mol_dict[ligand_path.stem]
else:
    mol = Chem.MolFromPDBFile(ligand_path)
    for atom in mol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info:
            atom_name = pdb_info.GetName().strip().upper()
            atom.SetProp("name", atom_name)
```

[TODO] We have implemented scoring_only that predicts affinity directly from input PDB without docking, but it has not been validated. We plan to officially support this feature in the future.

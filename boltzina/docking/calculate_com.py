import argparse
import sys
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

def get_center_of_mass_from_file(file_path):
    """
    Load molecule from file using RDKit and calculate center of mass.
    File format is automatically determined from the file extension.

    Args:
        file_path (str): Path to the molecular structure file

    Returns:
        numpy.ndarray: Center of mass coordinates [x, y, z]

    Raises:
        ValueError: If file format is unsupported or molecule cannot be loaded
        ValueError: If input file does not contain 3D coordinate information
    """
    # Get file extension
    ext = os.path.splitext(file_path)[1].lower()

    mol = None
    if ext == '.pdb':
        mol = Chem.MolFromPDBFile(file_path, removeHs=False)
    elif ext == '.sdf':
        # SDF files can contain multiple molecules, so get the first one
        suppl = Chem.SDMolSupplier(file_path, removeHs=False)
        if suppl:
            mol = next(suppl, None)
    elif ext == '.mol2':
        mol = Chem.MolFromMol2File(file_path, removeHs=False)
    else:
        # Try other common formats
        try:
            mol = Chem.MolFromMolFile(file_path, removeHs=False)
        except Exception:
            pass

    if mol is None:
        raise ValueError(f"Unsupported file format or failed to load molecule: {file_path}")

    # Check if molecule has 3D conformers
    if mol.GetNumConformers() == 0:
        raise ValueError("Input file does not contain 3D coordinate information.")

    # Calculate center of mass for the first conformer

    conformer = mol.GetConformer()
    center_of_mass = calculate_center_of_mass(mol, conformer)

    return center_of_mass

def calculate_center_of_mass(mol, conformer):
    """
    Calculate mass-weighted center of mass for a molecule conformer.

    Args:
        mol: RDKit molecule object
        conformer: RDKit conformer object

    Returns:
        numpy.ndarray: Mass-weighted center of mass coordinates [x, y, z]
    """
    total_mass = 0.0
    weighted_coords = np.zeros(3)

    # Iterate through all atoms
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_mass = atom.GetMass()
        atom_pos = conformer.GetAtomPosition(atom_idx)

        # Add mass-weighted coordinates
        weighted_coords[0] += atom_mass * atom_pos.x
        weighted_coords[1] += atom_mass * atom_pos.y
        weighted_coords[2] += atom_mass * atom_pos.z

        total_mass += atom_mass

    # Calculate center of mass
    center_of_mass = weighted_coords / total_mass

    return center_of_mass

def main():
    parser = argparse.ArgumentParser(description='Calculate center of mass for molecular files (RDKit version)')
    parser.add_argument('molecule_file', help='Input molecule file (PDB, SDF, MOL2, etc.)')
    parser.add_argument('--size_x', type=float, default=20, help='Box size X dimension (default: 20)')
    parser.add_argument('--size_y', type=float, default=20, help='Box size Y dimension (default: 20)')
    parser.add_argument('--size_z', type=float, default=20, help='Box size Z dimension (default: 20)')
    parser.add_argument('--output', default='input.txt', help='Output file name (default: input.txt)')

    args = parser.parse_args()

    try:
        # Calculate center of mass
        com = get_center_of_mass_from_file(args.molecule_file)

        # Write to output file
        with open(args.output, 'w') as f:
            f.write(f"center_x = {com.x:.3f}\n")
            f.write(f"center_y = {com.y:.3f}\n")
            f.write(f"center_z = {com.z:.3f}\n")
            f.write(f"size_x = {args.size_x}\n")
            f.write(f"size_y = {args.size_y}\n")
            f.write(f"size_z = {args.size_z}\n")

        print(f"Molecule center of mass: ({com.x:.3f}, {com.y:.3f}, {com.z:.3f})")
        print(f"Output written to: {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

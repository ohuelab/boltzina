import argparse
import sys
import os
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

def get_center_of_mass_from_file(file_path):
    """
    Load molecule from file using RDKit and calculate center of mass.
    File format is automatically determined from the file extension.
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
    conformer = mol.GetConformer(0)
    center_of_mass = rdMolTransforms.ComputeCenterOfMass(conformer)

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

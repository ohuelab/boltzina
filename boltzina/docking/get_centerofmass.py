import pymol
from pymol import cmd
import argparse
import sys

def get_center_of_mass(selection):
    """Calculate center of mass for a selection using PyMOL"""
    # Get the center of mass using PyMOL's built-in function
    com = cmd.centerofmass(selection)
    return com

def main():
    parser = argparse.ArgumentParser(description='Calculate center of mass for a chain in a CIF file')
    parser.add_argument('cif_file', help='Input CIF file path')
    parser.add_argument('--chain', required=True, help='Chain identifier (e.g., B)')
    parser.add_argument('--size_x', type=float, default=20, help='Box size in X dimension')
    parser.add_argument('--size_y', type=float, default=20, help='Box size in Y dimension')
    parser.add_argument('--size_z', type=float, default=20, help='Box size in Z dimension')
    parser.add_argument('--output', default='input.txt', help='Output file name')

    args = parser.parse_args()

    # Initialize PyMOL in quiet mode
    pymol.finish_launching(['pymol', '-c'])

    try:
        # Load the CIF file
        cmd.load(args.cif_file, 'structure')

        # Create selection for the specified chain
        chain_selection = f'structure and chain {args.chain}'

        # Check if the chain exists
        if cmd.count_atoms(chain_selection) == 0:
            print(f"Error: Chain {args.chain} not found in the structure")
            sys.exit(1)

        # Calculate center of mass
        com = get_center_of_mass(chain_selection)

        # Write output file
        with open(args.output, 'w') as f:
            f.write(f"center_x = {com[0]:.3f}\n")
            f.write(f"center_y = {com[1]:.3f}\n")
            f.write(f"center_z = {com[2]:.3f}\n")
            f.write(f"size_x = {args.size_x}\n")
            f.write(f"size_y = {args.size_y}\n")
            f.write(f"size_z = {args.size_z}\n")

        print(f"Center of mass for chain {args.chain}: ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
        print(f"Output written to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up PyMOL
        cmd.quit()

if __name__ == "__main__":
    main()

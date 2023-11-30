
from rdkit import Chem

smiles = 'C'

from openbabel import openbabel

def smiles_to_sdf(smiles, output_file):
    # Initialize Open Babel conversion
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "sdf")

    # Create a molecule object
    mol = openbabel.OBMol()

    # Read the SMILES string
    if not obConversion.ReadString(mol, smiles):
        raise ValueError("Invalid SMILES string")

    # Convert and save to SDF format
    obConversion.WriteFile(mol, output_file)

# Example usage
output_filename = "output.sdf"
smiles_to_sdf(smiles, output_filename)

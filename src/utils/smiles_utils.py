from rdkit import Chem
from openbabel import pybel

def canonize(smiles: str, raise_error: bool = True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        if(raise_error): raise e

def get_mol_size(smiles: str):
    mol = pybel.readstring('smi', smiles)
    mol.addh()
    return mol.OBMol.NumAtoms()

from rdkit import Chem
from openbabel import pybel

def canonize(smiles: str):    
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def get_mol_size(smiles: str):
    mol = pybel.readstring('smi', smiles)
    mol.addh()
    return mol.OBMol.NumAtoms()

from openbabel import pybel
import numpy as np
from utils import xyz
from . import covalent_radii

def get_bond_id(elem1, elem2, order):
    bond_descriptor = '-'.join(sorted([elem1, elem2])) # Ex: 'C-H' (alphabetical order)
    return f'{bond_descriptor} ({order})'

def _get_bond_matrix(mol):
    atoms_len = mol.NumAtoms()

    bonds_matrix = np.zeros((atoms_len, atoms_len), dtype=int)
    for bond in pybel.ob.OBMolBondIter(mol):
        idx1, idx2  = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        bonds_matrix[idx1, idx2] = 1
        bonds_matrix[idx2, idx1] = 1

    return bonds_matrix

def validate_comparing_smiles_vs_xyz_bonds(smiles, xyz_path):
    smiles_molecule = pybel.readstring('smi', smiles)
    smiles_molecule.addh()
    smiles_mol = smiles_molecule.OBMol

    xyz_mol = xyz.xyz2pybel(xyz_path)

    smiles_matrix = _get_bond_matrix(smiles_mol)
    xyz_matrix = _get_bond_matrix(xyz_mol)

    return np.array_equal(smiles_matrix, xyz_matrix) 

def validate_with_covalent_radiuses(smiles, xyz_path):
    smiles_molecule = pybel.readstring('smi', smiles)
    smiles_molecule.addh()
    mol = smiles_molecule.OBMol

    atoms_positions = xyz.read_xyz(xyz_path)['positions']

    bonds_broken = list()
    for bond in pybel.ob.OBMolBondIter(mol):
        idx1, idx2  = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        atom1, atom2 = atoms_positions[idx1], atoms_positions[idx2]
        position1, position2 = atom1['position'], atom2['position']
        elem1, elem2 = atom1['element'], atom2['element']
        bond_order = bond.GetBondOrder()
        bond_id = get_bond_id(elem1, elem2, bond_order)
        
        length = np.linalg.norm(position1 - position2)
        covalent_limit = covalent_radii.calc_covalent_limit(elem1, elem2)
        if length > covalent_limit:
            bonds_broken.append({
                'bond': bond_id,
                'length': length,
                'limit': covalent_limit,
            })
    
    return {
        'valid': len(bonds_broken) == 0,
        'bonds_broken': bonds_broken,
    }

def validate(smiles, xyz_path):
    return validate_with_covalent_radiuses(smiles, xyz_path)

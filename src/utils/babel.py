from openbabel import pybel

def create_mol_file_from_smiles(smile, path):
    mol = pybel.readstring('smi', smile)
    mol.addh()
    mol.make3D()
    mol.localopt()
    mol.write('xyz', str(path), overwrite=True)

    return mol

def smiles_to_mol(smiles):
    return pybel.readstring('smi', smiles)

def optimize_geometry(mol):
    if not mol: return

    mol.addh()
    mol.make3D()
    mol.localopt()
    return mol

def save_xyz(mol, path):
    if not mol: return
    mol.write('xyz', str(path), overwrite=True)
import numpy as np
from openbabel import pybel

def read_xyz(xyz_path):
    with open(xyz_path) as file:
        lines = file.readlines()
        length = int(lines[0])
        positions = [line.split() for line in lines[2:]] # filter lines with atoms positions
    xyz = [{'element': pos[0], 'position': np.array(pos[1:4]).astype(float)} for pos in positions] # split atom element from position
    return { 'length': length, 'positions': xyz }

def xyz2pybel(xyz_path):
    with open(xyz_path) as file:
        xyz_file = file.read()
        obconversion = pybel.ob.OBConversion()
        obconversion.SetInFormat('xyz')
        mol = pybel.ob.OBMol()
        obconversion.ReadString(mol, xyz_file)

    return mol
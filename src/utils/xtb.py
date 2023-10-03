import subprocess
from pathlib import Path
import os

xtb_path = Path(os.getcwd()) / '../executables/xtb/bin/xtb'

def check_magnetic_moment(logs: str):
    spin = -1
    for line in logs.split('\n'):
        if 'spin' in line:
            spin = float(line.split()[2])
            break
    
    return spin == 0

def check_geometry_convergence(logs: str):
    search_str = 'GEOMETRY OPTIMIZATION CONVERGED'
    converged = [line for line in logs.split('\n') if search_str in line]
    return len(converged) > 0

def check_distances(logs: str):
    lines = logs.split('\n')
    start = -1
    for i, line in enumerate(lines):
        if('selected distances' in line):
            start = i + 3
            count = int(line.split()[1])

    data = [float(item.split()[6]) for item in lines[start:start + count]]
    if max(data) < 3:
        return True

    return False

def check_hessian_matrix(logs: str):
    search_str = 'projected vibrational frequencies'
    lines = logs.split('\n')
    start = [i for i, line in enumerate(lines) if search_str in line][0] + 1

    i = start
    while('eigval' in lines[i]):
        eigenvalues = [float(value) for value in lines[i].split()[2:]]
        if(min(eigenvalues) < 0):
            return False
        i += 1

    return True

class XTB_process():
    def __init__(self, process_path='./'):
        self.process_path = process_path

    def run_process(self, command):
        if isinstance(command, str): command = command.split()
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.process_path)
        return process.stdout.decode('UTF-8')

    def process_spin(self, mol_path: str, charge):
        # command = f'{xtb_path} {mol_path} --sp --chrg {charge}' # generating molecule spin with xtb
        command = [xtb_path, mol_path, '--sp', '--chrg', str(charge)] # generating molecule spin with xtb
        return self.run_process(command)

    def process_geometry_optimization(self, mol_path: str, charge):
        command = f'{xtb_path} {mol_path} --chrg {charge} --opt verytight' # optimizing molecule with xtb
        return self.run_process(command)

    def process_hessian_matrix(self, optimized_mol_path, charge):
        command = f'{xtb_path} {str(optimized_mol_path)} --hess --chrg {charge}' # optimizing molecule with xtb
        return self.run_process(command)

    def delete_xtb_files(self):
        files = [
            'wbo',
            'xtbopt.log',
            'xtbrestart',
            'xtbtopo.mol',
            'charges',
            'xtblast.xyz',
            '.xtboptok',
            'NOT_CONVERGED',
            '.sccnotconverged',
            'vibspectrum',
            'g98.out',
            'hessian',
        ]

        for filename in files:
            file = self.process_path / Path(filename)
            if file.is_file(): file.unlink()


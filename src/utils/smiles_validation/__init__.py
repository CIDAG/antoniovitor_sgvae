from rdkit import Chem
from utils import xtb, babel
from pathlib import Path
import shutil
import os
from . import bonds_validation
from utils.timer import timer
from tqdm import tqdm

########## FUNCTIONS
def save_file(path, data):
    with open(path, 'w') as file: # saving spin file
        file.write(data)

def log_states(states):
    remaining = len([i for i in states if not has_error(i)])
    return { 'input': remaining, 'percentual': remaining/len(states) }

def has_error(state):
    return 'error' in state

########## FILENAMES
ff_filename = 'forcefieldopt.xyz'
xtb_filename = 'xtbopt.xyz'

########## CRASH PREVENTION
def set_error_on_crash(state, *args, **kwargs):
    state['error'] = 'UNKNOWN_ERROR'
    return state

def log_on_crash(func):
    def applicator(*args, **kwargs):
        return func(*args,**kwargs)
        try:
            return func(*args,**kwargs)
        except Exception as e:
            return set_error_on_crash(*args, **kwargs)
    return applicator

########## VALIDATION
@log_on_crash
def check_if_smiles_exists(state):
    if has_error(state): return state

    if not state['smiles']: state['error'] = 'SMILE_DOES_NOT_EXIST'

    return state

@log_on_crash
def convert_to_mol_object(state):
    if has_error(state): return state

    mol = Chem.MolFromSmiles(state['smiles'])
    if not mol:
        state['error'] = 'RDKIT_CAN_NOT_READ_SMILES'
    
    state['mol'] = mol
    return state

@log_on_crash
def convert_to_canonical_smiles(state):
    if has_error(state): return state

    canonical_smiles = Chem.MolToSmiles(state['mol'], isomericSmiles=True, canonical=True)
    if not canonical_smiles:
        state['error'] = 'CAN_NOT_CREATE_CANONICAL_SMILES'

    state['canonical'] = canonical_smiles
    return state

@log_on_crash
def set_variables(state, id, save_path):
    if has_error(state): return state

    state['id'] = id
    state['paths'] = dict()

    save_path = save_path / str(id)
    save_path.mkdir(parents=True, exist_ok=True)
    state['paths']['save'] = save_path

    return state

@log_on_crash
def check_if_smiles_already_exists(state, existing_smiles: list):
    if has_error(state): return state

    if state['canonical'] in existing_smiles:
        state['error'] = 'SMILES_ALREADY_EXISTS_IN_DATASET'

    return state

@log_on_crash
def check_if_smiles_was_recreated(states: list):
    smiles_set = set()
    for state in states:
        if has_error(state): continue

        canonical = state['canonical']
        if canonical in smiles_set:
            state['error'] = 'SMILES_SAMPLED_REPEATEDLY'
        else:
            smiles_set.add(canonical)

    return states

@log_on_crash
def check_smiles_charge(state):
    if has_error(state): return state

    # check if charge was given
    if 'charge' not in state: return state

    formal_charge = Chem.rdmolops.GetFormalCharge(state['mol'])

    if state['charge'] != formal_charge:
        state['error'] = 'FORMAL_CHARGE_OF_IONS_IS_NOT_1'
    elif state['charge'] < 0 and '-' not in state['canonical']: 
        state['error'] = 'IONS_DOES_NOT_HAVE_FORMAL_CHARGE'
    elif state['charge'] > 0 and '+' not in state['canonical']:
        state['error'] = 'IONS_DOES_NOT_HAVE_FORMAL_CHARGE'

    return state

@log_on_crash
def set_formal_charge(state):
    if has_error(state): return state

    if 'charge' not in state:
        state['charge'] = Chem.rdmolops.GetFormalCharge(state['mol'])

    return state

@log_on_crash
def optimize_geometry_with_force_field(state):
    if has_error(state): return state

    state['paths']['force_field'] = state['paths']['save'] / ff_filename

    mol = babel.smiles_to_mol(state['canonical'])
    mol = babel.optimize_geometry(mol)
    babel.save_xyz(mol, state['paths']['force_field']) 

    if not mol:
        state['error'] = 'OPENBABEL_CAN_NOT_OPTIMIZE_SMILES'

    return state

@log_on_crash
def check_bonds_with_force_field_opt(state):
    if has_error(state): return state
    canonical = state['canonical']
    ff_path = state['paths']['force_field']

    validation = bonds_validation.validate(canonical, ff_path)
    if(not validation['valid']):
        state['error'] = 'FORCE_FIELD_OPT_WITH_BROKEN_BONDS'

    return state

@log_on_crash
def check_spin(state):
    if has_error(state): return state
    paths = state['paths']
    
    xtb_process = xtb.XTB_process(process_path=paths['save'])
    state['xtb'] = xtb_process

    spin_log = xtb_process.process_spin(ff_filename, state['charge']) # checking magnetic moment
    save_file(paths['save'] / f'spin.log', spin_log) # saving spin file

    if not xtb.check_magnetic_moment(spin_log): # XTB spin validation
        state['error'] = 'SPIN_IS_NOT_ZERO'

    return state

@log_on_crash
def optimize_geometry_with_xtb(state):
    if has_error(state): return state
    
    # checking XTB geometry optimization
    optimization_log = state['xtb'].process_geometry_optimization(ff_filename, state['charge'])
    save_file(state['paths']['save'] / f'optimization.log', optimization_log) # saving optimization file

    if not xtb.check_geometry_convergence(optimization_log):
        state['error'] = 'GEOMETRY_DID_NOT_CONVERGE'
        return state
    if not xtb.check_distances(optimization_log):
        state['error'] = 'BONDS_HAVE_MORE_THAN_3_ANGSTROMS'
        return state

    return state

@log_on_crash
def check_bonds_with_xtb_opt(state):
    if has_error(state): return state

    state['paths']['xtb'] = state['paths']['save'] / xtb_filename

    
    validation = bonds_validation.validate(state['canonical'], state['paths']['xtb'])
    if(not validation['valid']):
        state['error'] = 'XTB_OPT_WITH_BROKEN_BONDS'

    return state

@log_on_crash
def check_hessian_matrix_eigenvalues(state):
    if has_error(state): return state

    hessian_log = state['xtb'].process_hessian_matrix(xtb_filename, state['charge'])
    save_file(state['paths']['save'] / f'hessian.log', hessian_log)

    if not xtb.check_hessian_matrix(hessian_log):
        state['error'] = 'HESSIAN_MATRIX_HAVE_NEGATIVES_EIGENVALUES'

    return state

@log_on_crash
def save_error(state):
    if has_error(state) and 'paths' in state:
        save_file(state['paths']['save'] / f'error.log', state['error'])
    return { **state, 'result': True}

@log_on_crash
def rename_smiles_folder(state, save_path: Path):
    if ('paths' not in state): return state
    if (state.get('error') == 'SMILES_SAMPLED_REPEATEDLY'):
        shutil.rmtree(state['paths']['save'], ignore_errors=True)
        return state

    dest_path = save_path / state['canonical']
    if(dest_path.exists()): shutil.rmtree(dest_path)
    shutil.move(state['paths']['save'], dest_path)
    return state

_filtered_keys = ['xtb', 'mol', 'id', 'paths']
@log_on_crash
def filter_final_state(state: dict):
    for key in _filtered_keys: state.pop(key, None)

    return state

def set_results(state):
    if has_error(state): return { **state, 'valid': False }
    return { **state, 'valid': True}

def validate_smiles(smiles_list: list, existing_smiles: set, save_base_path, charge = None, logs_path = Path('logs.csv')):
    save_base_path = Path(save_base_path)
    states = [{ 'smiles': smi } for smi in smiles_list]

    if (charge is not None):
        states = [{ **state, 'charge': charge } for state in states]

    with timer(logs_path, 'check_if_smiles_exists',                log_states(states)): states = [check_if_smiles_exists(state)                 for state in tqdm(states, desc='check_if_smiles_exists')]
    with timer(logs_path, 'convert_to_mol_object',                 log_states(states)): states = [convert_to_mol_object(state)                  for state in tqdm(states, desc='convert_to_mol_object')]
    with timer(logs_path, 'convert_to_canonical_smiles',           log_states(states)): states = [convert_to_canonical_smiles(state)            for state in tqdm(states, desc='convert_to_canonical_smiles')]
    with timer(logs_path, 'set_variables',                         log_states(states)): states = [set_variables(state, i, save_base_path)       for i, state in tqdm(enumerate(states), desc='set_variables')]
    with timer(logs_path, 'check_if_smiles_already_exists',        log_states(states)): states = [check_if_smiles_already_exists(state, existing_smiles)    for state in tqdm(states, desc='check_if_smiles_already_exists')]
    with timer(logs_path, 'check_if_smiles_was_recreated',         log_states(states)): states =  check_if_smiles_was_recreated(tqdm(states, desc='check_if_smiles_was_recreated'))
    with timer(logs_path, 'check_smiles_charge',                   log_states(states)): states = [check_smiles_charge(state)                    for state in tqdm(states, desc='check_smiles_charge')]
    with timer(logs_path, 'set_formal_charge',                     log_states(states)): states = [set_formal_charge(state)                      for state in tqdm(states, desc='set_formal_charge')]
    with timer(logs_path, 'optimize_geometry_with_force_field',    log_states(states)): states = [optimize_geometry_with_force_field(state)     for state in tqdm(states, desc='optimize_geometry_with_force_field')]
    with timer(logs_path, 'check_bonds_with_force_field_opt',      log_states(states)): states = [check_bonds_with_force_field_opt(state)       for state in tqdm(states, desc='check_bonds_with_force_field_opt')]
    with timer(logs_path, 'check_spin',                            log_states(states)): states = [check_spin(state)                             for state in tqdm(states, desc='check_spin')]
    with timer(logs_path, 'optimize_geometry_with_xtb',            log_states(states)): states = [optimize_geometry_with_xtb(state)             for state in tqdm(states, desc='optimize_geometry_with_xtb')]
    with timer(logs_path, 'check_bonds_with_xtb_opt',              log_states(states)): states = [check_bonds_with_xtb_opt(state)               for state in tqdm(states, desc='check_bonds_with_xtb_opt')]
    with timer(logs_path, 'check_hessian_matrix_eigenvalues',      log_states(states)): states = [check_hessian_matrix_eigenvalues(state)       for state in tqdm(states, desc='check_hessian_matrix_eigenvalues')]
    with timer(logs_path, 'save_error',                            log_states(states)): states = [save_error(state)                             for state in tqdm(states, desc='save_error')]
    with timer(logs_path, 'rename_smiles_folder',                  log_states(states)): states = [rename_smiles_folder(state, save_base_path)   for state in tqdm(states, desc='rename_smiles_folder')]
    with timer(logs_path, 'filter_final_state',                    log_states(states)): states = [filter_final_state(state)                     for state in tqdm(states, desc='filter_final_state')]
    with timer(logs_path, 'set_results',                           log_states(states)): states = [set_results(state)                            for state in tqdm(states, desc='set_results')]

    return states

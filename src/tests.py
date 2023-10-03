from utils import smiles_validation
from pathlib import Path

base_path = Path('temp')
charge = None
existing_smiles = set('C')
smiles_list = [
    'C',
    'CC'
]

results = smiles_validation.validate_smiles_in_batches(
    smiles_list,
    existing_smiles,
    base_path,
    charge,
)

print(results)
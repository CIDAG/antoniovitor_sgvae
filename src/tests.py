import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from grammar import Grammar

grammar = Grammar()


base_path = Path('../datasources/electrolyte')
df = pd.read_csv(base_path / 'dirty_data.csv')

errors = []
for smi in df['smiles']:
    try:
        grammar.parse_smiles_list([smi])
        errors.append('')
    except Exception as e:
        errors.append(str(e))
        print(smi, ' | ', e)

df['errors'] = errors

df
from pathlib import Path
import pandas as pd
from db import MongoDB
import utils.smiles_utils as smiles_utils
from torch.utils.data import Dataset

base_path = Path('../datasources/IL_ESW')
base_path.mkdir(parents=True, exist_ok=True)

df_path = base_path / 'dataset_complete'

anions_df = pd.read_csv(df_path / f'anions.csv')
cations_df = pd.read_csv(df_path / f'cations.csv')

anions_df = anions_df.iloc[:10]
cations_df = cations_df.iloc[:10]

df = anions_df.merge(cations_df, how='cross', suffixes=['_anions','_cations'])
df['esw'] = df[['lumo-fopt_anions', 'lumo-fopt_cations']].min(axis=1) - df[['homo-fopt_anions', 'homo-fopt_cations']].max(axis=1)

df.to_csv()



class ESW():
    pass
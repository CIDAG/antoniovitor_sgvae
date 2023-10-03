from pathlib import Path
import pandas as pd
import numpy as np
from db import MongoDB
import utils.smiles_utils as smiles_utils
from torch.utils.data import Dataset, Subset
from datasource_manager import IL_ESW_Extended_Datasource
import random
import pymongo
import grammar.old_grammar as grammar
from tqdm import tqdm
import time
import constants as c

class ESW_IL_Extended_Dataset(Dataset):
    def __init__(self, subset, transform=None) -> None:
        super().__init__()
        self.dataset_name = f'IL_ESW'
        self.subset = subset
        self.transform = transform

        IL_ESW_Extended_Datasource().download_IL_ESW_extended()
        self.df = self._load_dataframe()

    def set_transform(self, transform):
        self.transform=transform
    
    def _load_dataframe(self) -> pd.DataFrame:
        dataset_path = IL_ESW_Extended_Datasource.datasource_path
        cache_path = dataset_path / '__cache__' / f'{self.subset}.parquet'
        cache_path.parent.mkdir(exist_ok=True) # creates directory


        if(cache_path.exists()):
            return pd.read_parquet(cache_path)
        
        df = pd.read_csv(dataset_path / f'{self.subset}.csv')
        df = df.rename(columns={'homo-fopt': 'homo', 'lumo-fopt': 'lumo', 'gap-fopt': 'gap', 'e_tot-fopt': 'e_total'})
        df.to_parquet(cache_path)

        return df
    
    def _save_dataframe(self):
        cache_path = IL_ESW_Extended_Datasource.datasource_path / '__cache__' / f'{self.subset}.feather'
        self.df.to_parquet(cache_path)

    def parse_smiles(self, fields: list = ['smiles']):
        for field in fields:
            smiles_list = self.df[field].to_list()

            print(f'Parsing smiles at field {field} in dataset {self.dataset_name}')
            self.df[f'{field}_ohe'] = grammar.parse_smiles_list(smiles_list, verbose=True)
        self._save_dataframe()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return self.transform(item) if(self.transform) else item

class ESW_IL_Extended_Subset(Dataset):
    def __init__(self, subset, transform=None) -> None:
        super().__init__()
        self.dataset_name = f'IL_ESW'
        self.subset = subset
        self.transform = transform

        IL_ESW_Extended_Datasource().download_IL_ESW_extended()
        self.df = self._load_dataframe()
        self.df = self.df.sample(n=100000, random_state=c.seed)

    def set_transform(self, transform):
        self.transform=transform
    
    def _load_dataframe(self) -> pd.DataFrame:
        dataset_path = IL_ESW_Extended_Datasource.datasource_path
        cache_path = dataset_path / '__cache__' / f'{self.subset}.parquet'
        cache_path.parent.mkdir(exist_ok=True) # creates directory


        if(cache_path.exists()):
            return pd.read_parquet(cache_path)
        
        df = pd.read_csv(dataset_path / f'{self.subset}.csv')
        df = df.rename(columns={'homo-fopt': 'homo', 'lumo-fopt': 'lumo', 'gap-fopt': 'gap', 'e_tot-fopt': 'e_total'})
        df.to_parquet(cache_path)

        return df

    def parse_smiles(self, fields: list = ['smiles']):
        for field in fields:
            smiles_list = self.df[field].to_list()

            print(f'Parsing smiles at field {field} in dataset {self.dataset_name}')
            self.df[f'{field}_ohe'] = grammar.parse_smiles_list(smiles_list, verbose=True)
        self._save_dataframe()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return self.transform(item) if(self.transform) else item

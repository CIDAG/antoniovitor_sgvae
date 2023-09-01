from pathlib import Path
import pandas as pd
import numpy as np
from db import MongoDB
import utils.smiles_utils as smiles_utils
from torch.utils.data import Dataset
from datasource_manager import IL_ESW_Datasource
import random
import pymongo
import grammar
from tqdm import tqdm
import time

class ESW_IL_Dataset(Dataset):
    def __init__(self, subset, transform=None) -> None:
        super().__init__()
        self.dataset_name = f'IL_ESW'
        self.subset = subset
        self.transform = transform

        IL_ESW_Datasource().download_IL_ESW_repository()
        self.df = self._load_dataframe()

    def set_transform(self, transform):
        self.transform=transform
    
    def _load_dataframe(self) -> pd.DataFrame:
        dataset_path = IL_ESW_Datasource.datasource_path
        cache_path = dataset_path / '__cache__' / f'{self.subset}.parquet'
        cache_path.parent.mkdir(exist_ok=True) # creates directory

        if(cache_path.exists()):
            return pd.read_parquet(cache_path)
        
        df = pd.read_csv(dataset_path / 'dataset_complete' / f'{self.subset}.csv')
        df = df.rename(columns={'homo-fopt': 'homo', 'lumo-fopt': 'lumo', 'gap-fopt': 'gap', 'e_tot-fopt': 'e_total'})
        df['canonical'] = [smiles_utils.canonize(smiles) for smiles in df['smiles']]
        df.to_parquet(cache_path)

        return df
    
    def _save_dataframe(self):
        cache_path = IL_ESW_Datasource.datasource_path / '__cache__' / f'{self.subset}.feather'
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


class IonicLiquidsMongoDataset(Dataset):
    datasource_path = IL_ESW_Datasource.datasource_path
    _indexes: list

    def __init__(self, subset, transform=None) -> None:
        super().__init__()

        self.dataset_name = f'IL_ESW'
        self.subset = subset
        self.collection_name = f'dataset_{self.dataset_name}_{self.subset}'
        self.transform = transform

        self.db = MongoDB().get_db()
        self.collection = self.db[self.collection_name]

        IL_ESW_Datasource().download_IL_ESW_repository()
        self._process()

        self._indexes = [i['_id'] for i in self.collection.find({}, { '_id': 1,  'mol_id': 1}).sort('mol_id')]

    def set_transform(self, transform):
        self.transform=transform

    def seed(self, seed = 42):
        random.seed(seed)

    def shuffle(self):
        self._indexes = random.sample(self._indexes, len(self._indexes))

    def parse_smiles(self):
        records = list(self.collection.find({ "x" : { "$exists" : False } }, { '_id': 1, 'smiles': 1, 'canonical': 1 }).sort('mol_id'))
        if(len(records) == 0): return # All dataset is processed

        smiles_list = [record['smiles'] for record in records]
        x_list = grammar.parse_smiles_list(smiles_list, verbose=True)
        for record, x in tqdm(zip(records, x_list), desc='Saving parsed SMILES'):
            self.collection.update_one({'_id': record['_id']}, { '$set': { 'x': x.tolist() } })

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx):
        if(type(idx) == list):
            _ids = [self._indexes[i] for i in idx]
            records = self._get_records(_ids)
            return [self._process_record(i) for i in records]

        if isinstance(idx, slice):
            _ids = self._indexes[idx]
            records = self._get_records(_ids)
            return [self._process_record(i) for i in records]
        
        record = self._get_record(self._indexes[idx])
        return self._process_record(record) 


    def _process_record(self, record):
        if(self.transform):
            return self.transform(record)
        return record

    def _get_records(self, _ids):
        return self._get_collection().find({'_id': {'$in': _ids}})

    def _get_record(self, _id):
        return self._get_collection().find_one({'_id': _id})
    
    def _get_collection(self):
        return self.db[self.collection_name]

    def _process(self):
        # checking if dataset is already processed
        if(self.collection_name in self.db.list_collection_names()): return

        print('Processing dataset... ', end='')
        df = pd.read_csv(self.datasource_path / 'dataset_complete' / f'{self.subset}.csv')
        df = df.rename(columns={'homo-fopt': 'homo', 'lumo-fopt': 'lumo', 'gap-fopt': 'gap', 'e_tot-fopt': 'e_total'})
        df['canonical'] = [smiles_utils.canonize(smiles) for smiles in df['smiles']]

        data = df[['mol_id', 'smiles', 'canonical', 'homo', 'lumo', 'gap', 'e_total']].to_dict('records')

        self.collection.drop()
        self.collection.create_indexes([
            pymongo.IndexModel('smiles', unique=True),
            pymongo.IndexModel('canonical', unique=True),
        ])

        self.collection.insert_many(data)
        print('Done')

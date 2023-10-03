from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json

class ProcessedDataset(Dataset):
    def __init__(self, dataset_name, transform=None) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dir = Path('../datasets') / self.dataset_name
        self.transform = transform

        with open(self.dir / 'dataset_info.json') as file:
            self.info = json.load(file)

        self.df = pd.read_parquet(self.dir / f'{self.info["filename"]}.parquet')
        
    def set_transform(self, transform):
        self.transform=transform
    
    def to_df(self):
        return self.df
    
    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return self.transform(item) if(self.transform) else item
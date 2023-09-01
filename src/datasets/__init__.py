from datasets.IL_ESW import ESW_IL_Dataset
from datasets.IL_ESW_extended import ESW_IL_Extended_Dataset, ESW_IL_Extended_Subset
from . import transforms
import numpy as np
from torch.utils.data import Subset

def select(dataset_name: str, subset:str = None):
    if(dataset_name == 'IL_ESW'):
        return ESW_IL_Dataset(subset=subset)
    elif(dataset_name == 'IL_ESW_extended'):
        return ESW_IL_Extended_Dataset(subset=subset)
    elif(dataset_name == 'IL_ESW_extended_subset'):
        return ESW_IL_Extended_Subset(subset=subset)


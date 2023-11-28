import click
import math
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit import RDLogger
from pathlib import Path
from utils import smiles_validation
from models.sgvae import SGVAE
import random
import constants as c
from datasets.processed_dataset import ProcessedDataset

from torch.utils.data import random_split
from torch import Generator


RDLogger.DisableLog('rdApp.*')

def chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i + n]


# TODO: update pytorch and use proportions directly in random_split
def calc_sets_splits_sizes(length, proportions: list):
    if(sum(proportions) != 1): raise ValueError('Sum of proportions is not 1 during split operation')


    splits_sizes = [math.floor(percentual*length) for percentual in proportions[:-1]]
    splits_sizes.append(length - sum(splits_sizes)) # add remains in the last split 

    return splits_sizes

@click.command()
@click.option(
    '--num_tries',
    type=click.INT,
    required=True,
)
@click.option(
    '--dataset_name',
    type=click.STRING,
    prompt='Dataset name',
    required=True,
)
@click.option(
    '--model_path',
    prompt='Model path',
    type=click.Path(exists=True, path_type=Path),
    required=True
)
@click.option(
    '--should_restart',
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option(
    '--property',
    type=click.STRING,
)
@click.option(
    '--target',
    type=click.FLOAT,
)
def run(num_tries, dataset_name, model_path,should_restart, property, target):
    # SEEDING
    np.random.seed(c.seed)
    random.seed(c.seed)

    # DATASET
    dataset = ProcessedDataset(dataset_name)
    dataset_df = dataset.to_df()


    sets_sizes = calc_sets_splits_sizes(length=len(dataset), proportions=[.07,.07, .86])
    sets = random_split(dataset, sets_sizes, generator=Generator().manual_seed(c.seed)) # splits datasets randomly
    test_set, validation_set, train_set = sets

    # VARIABLES
    subset = dataset.info.get('subset')
    existing_smiles = set(dataset_df['canonical'].to_list())

    charge = None
    if subset is not None:
        charge = -1 if subset == 'anions' else 1
    
    # MODEL
    sgvae = SGVAE().to(c.device)
    sgvae.load(model_path / 'best_models')
    sgvae.eval()

    # PATHS
    sampling_path = model_path / 'sampling'
    sampling_path.mkdir(parents=True, exist_ok=True)

    data_dir = sampling_path / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    logs_path = sampling_path / 'logs.csv'


    # UTILS
    filename = f'samples_without_validation'
    df_path = sampling_path / f'{filename}.pkl'
    def save_df(df: pd.DataFrame):
        df.to_pickle(df_path)
        df.drop(columns=['z']).to_csv(sampling_path / f'{filename}.csv')

    if should_restart or not df_path.exists():
        # TODO: check if target property is selected
        if (property is not None and target is not None):
            mi, ma = target - 0.1, target + 0.1
            selected_set = [i for i in train_set if mi <= i[property] <= ma]
        else:
            selected_set = [i for i in train_set]

        selected_set = random.choices(selected_set, k=num_tries)
        selected_smiles = [i['smiles'] for i in selected_set]
        selected_canonical = [i['canonical'] for i in selected_set]

        z = sgvae.encode(list(selected_smiles)).cpu().detach().tolist()
        z_tensor = torch.Tensor(z).to(c.device)
        
        with  torch.no_grad():
            samples = np.concatenate([sgvae.decode(chunk) for chunk in chunks(z_tensor, 1000)])
        
        df = pd.DataFrame({
            'samples': samples,
            'sampled_from': selected_canonical,
            'z':z,
        })
        df.index.rename('id')
        save_df(df)

    df = pd.read_pickle(df_path)
    samples = df['samples']

    results = smiles_validation.validate_smiles(
        smiles_list=samples,
        existing_smiles=existing_smiles,
        save_base_path=data_dir,
        charge=charge,
        logs_path=logs_path,
    )

    results_df = pd.DataFrame(results)
    results_df.to_pickle(sampling_path / 'validation_results.pkl')
    results_df.to_csv(sampling_path / 'validation_results.csv', index=False)

    valid_df = results_df[results_df['valid'] == True]
    valid_df.to_pickle(sampling_path / 'samples.pkl')
    valid_df.to_csv(sampling_path / 'samples.csv', index=False)


import click
from pathlib import Path
import pandas as pd
from utils import smiles_utils
import json
from tqdm import tqdm

def canonize_smiles(df):
    return df

def sample_dataset(df: pd.DataFrame, num_samples):
    if(len(df.index) < num_samples):
        print(f'[WARNING] --num_samples ignored: DataFrame length ({len(df.index)}) is less than samples required ({num_samples}.)')
        return  df

    print(f'[SAMPLING] Selected {num_samples} items from datasource.')
    return df.sample(n=num_samples)

@click.command()
@click.option(
    '--path',
    prompt='Dataset path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True
)
@click.option(
    '--dataset_name',
    type=click.STRING,
    prompt='Dataset name',
    required=True,
)
@click.option( # TODO: see how to display prompt for tuple types
    '--properties_fields',
    type=(str, str),
    # prompt='Properties fields',
    default=('homo', 'lumo'),
    required=True
)
@click.option(
    '--smiles_field',
    type=click.STRING,
    prompt='SMILES field',
    prompt_required=False,
    default='smiles'
)
@click.option(
    '--ion_type',
    type=click.STRING,
)
@click.option(
    '--num_samples',
    type=click.INT,
)
def run(path, dataset_name, smiles_field, properties_fields, ion_type, num_samples):
    print(f'Processing dataset {dataset_name}')

    df = pd.read_csv(path)
    df = df[[smiles_field, *properties_fields]] # filter only relevant columns

    # CLEANING DATA
    initial_length = len(df.index)
    print(f'[CLEANING] Initial length: {initial_length}')
    
    # DROP NAN VALUES
    df = df.dropna()

    if(len(df.index) < initial_length):
        loss_size = initial_length - len(df.index)
        print(f'\t[DROP NAN VALUES]: {loss_size} items removed | REMAINING: {len(df.index)}')
    
    # SAMPLING
    if(num_samples is not None): df = sample_dataset(df, num_samples)

    # CLEANING DATA AFTER CANONIZATION
    initial_length = len(df.index)

    df.insert(0, 'canonical',
        [smiles_utils.canonize(smi, raise_error=False) for smi in tqdm(df[smiles_field], desc='Canonization', leave=False)]
    )
    df = df.dropna()

    if(len(df.index) < initial_length):
        loss_size = initial_length - len(df.index)
        print(f'\t[CANONIZATION]: {loss_size} items removed | REMAINING: {len(df.index)}')

    # PARSING SMILES
    # TODO: parse smiles

    print(f'[CLEANING] Final length: {len(df.index)}')

    df.index.name = 'id'

    save_dir = Path('../datasets') / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = '_'.join([dataset_name, *properties_fields])

    # TODO: add max_rule_size 

    info = {
        'dataset_name': dataset_name,
        'ion_type': ion_type,
        'smiles_field': smiles_field,
        'properties_fields': properties_fields,
        'filename': filename,
    }
    with open(save_dir / 'dataset_info.json', 'w') as file:
        file.write(json.dumps(info, indent = 4) )

    df.to_csv(save_dir / f'{filename}.csv')
    df.to_parquet(save_dir / f'{filename}.parquet')
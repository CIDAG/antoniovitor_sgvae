import click
from pathlib import Path
import pandas as pd
import numpy as np
from utils import smiles_utils
import json
from tqdm import tqdm

def canonize_smiles(df: pd.DataFrame, num_samples: int):
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
    '-d',
    type=click.STRING,
    prompt='Dataset name',
    required=True,
)
@click.option( # TODO: see how to display prompt for tuple types
    '--properties_fields',
    '-p',
    multiple=True,
    default=['homo', 'lumo'],
    show_default=True,
    required=True
)
@click.option(
    '--smiles_field',
    '-s',
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
    print(f'[PREPROCESSING] Processing {dataset_name}')

    df = pd.read_csv(path)
    df = df[[smiles_field, *properties_fields]] # filter only relevant columns
    df = df.sample(frac=1) # shuffle dataset

    # CLEANING DATA
    initial_length = len(df.index)
    print(f'[CLEANING] Initial length: {initial_length}')
    
    # DROP NAN VALUES
    df = df.dropna()

    if(len(df.index) < initial_length):
        loss_size = initial_length - len(df.index)
        print(f'\t[CLEANING] Drop NaN values: {loss_size} items removed | Remaining: {len(df.index)}')

    # CANONIZATION
    initial_length = len(df.index)
    df['canonical'] = np.nan
    df['errors'] = np.nan

    valid = 0
    max_length = num_samples if(num_samples is not None) else len(df.index) # sets max_length (if None) as size of df

    for id, row in df.iterrows():
        smiles = row[smiles_field]
        try:
            df.loc[id, 'canonical'] = smiles_utils.canonize(smiles)
            valid += 1
            if valid >= max_length: break
        except Exception as e:
            df['errors'] = e

    errors_df = df[df['errors'].notna()]
    df = df.drop('errors', axis=1).dropna()

    if(len(df.index) < initial_length):
        loss_size = initial_length - len(df.index)
        print(f'\t[CANONIZATION]: {loss_size} items removed | REMAINING: {len(df.index)}')

    # PARSING SMILES
    # TODO: parse smiles

    print(f'[PREPROCESSING] Final length: {len(df.index)}')

    df.index.name = 'id'

    save_dir = Path('../datasets') / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = '_'.join([dataset_name, *properties_fields])

    info = {
        'dataset_name': dataset_name,
        'ion_type': ion_type,
        'smiles_field': smiles_field,
        'properties_fields': properties_fields,
        'filename': filename,
    }
    with open(save_dir / 'dataset_info.json', 'w') as file:
        file.write(json.dumps(info, indent = 4) )

    errors_df.to_csv(save_dir / f'{filename}..errors.csv')
    errors_df.to_parquet(save_dir / f'{filename}..errors.parquet')
    
    df.to_csv(save_dir / f'{filename}.csv')
    df.to_parquet(save_dir / f'{filename}.parquet')

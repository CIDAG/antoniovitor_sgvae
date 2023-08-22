import argparse
import torch

# TODO: remove properties which will be loaded through args and cached parameters

training_parameters = {
    'dataset_name': 'IL_ESW_extended_subset',
}

_default_parameters = {
    'dataset_name': 'IL_ESW_extended_subset',
    'subset': 'anions',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_length': 492,
    'latent_dim': 256,
    'properties_names': ['homo', 'lumo'],
    'learning_rate': 1e-3,
    'patience': 15,
    'min_lr': 1e-5,
    'prediction_weight': 1,
    'reconstruction_weight': 1,
    'kl_weight': 1,
    'epochs': 100,
}

def load_parameters():
    return _default_parameters
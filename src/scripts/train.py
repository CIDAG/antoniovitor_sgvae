import parameters_parser
import datasets
from datasets.processed_dataset import ProcessedDataset
from datasets.transforms import SGVAEDataFormatter
from models.sgvae import SGVAE
from torch.utils.data import DataLoader, random_split, BatchSampler, RandomSampler
from torch import Generator
import numpy as np
import math
import constants as c
import torch
import time
from datetime import datetime
from registry import FileRegistry, CSVRegistry, CombineRegistry
from torch.utils.tensorboard import SummaryWriter 
import multiprocessing
import click

from pathlib import Path

num_workers = multiprocessing.cpu_count() - 1

# TODO: update pytorch and use proportions directly in random_split
def calc_sets_splits_sizes(length, proportions: list):
    if(sum(proportions) != 1): raise ValueError('Sum of proportions is not 1 during split operation')


    splits_sizes = [math.floor(percentual*length) for percentual in proportions[:-1]]
    splits_sizes.append(length - sum(splits_sizes)) # add remains in the last split 

    return splits_sizes

"""
Creates a DataLoader which loads batches from dataset instead of a item
"""
def create_dataloader(dataset, batch_size = 64):
    batch_sampler = BatchSampler(RandomSampler(range(len(dataset))), batch_size=batch_size, drop_last=True)
    loader = DataLoader(
        dataset, sampler=batch_sampler, num_workers=num_workers, pin_memory=True,
        collate_fn=lambda x : tuple(torch.Tensor(i) for i in x[0])
    )
    return loader

def create_registries(save_dir, name):
    file_registry = FileRegistry(save_dir / f'{name}.vae.log')
    csv_registry = CSVRegistry(save_dir / f'{name}.vae.csv')
    return CombineRegistry([file_registry, csv_registry])

def run(dataset_name, batch_size):
    print('Starting training...\n')

    # PARAMETERS
    # TODO: remove parameters that could be constants 
    params = parameters_parser.load_parameters()
    epochs = params['epochs']

    prediction_weight = params['prediction_weight']
    reconstruction_weight = params['reconstruction_weight']
    kl_weight = params['kl_weight']

    device = params['device']
    learning_rate = params['learning_rate']
    patience = params['patience']
    min_lr = params['min_lr']

    # PATHS
    training_hash = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    
    saving_path = Path('../results') / dataset_name / training_hash
    saving_path.mkdir(parents=True, exist_ok=True)

    # LOGGING
    writer = SummaryWriter(log_dir=saving_path / 'logs')
    train_registry = create_registries(saving_path, name='train')
    validation_registry = create_registries(saving_path, name='validation')

    # DATASET
    dataset = ProcessedDataset(dataset_name)
    properties = dataset.info['properties_fields']
    dataset.set_transform(
        SGVAEDataFormatter(input_field='smiles', output_field='smiles', properties=properties)
    )

    # SPLIT SETS
    sets_sizes = calc_sets_splits_sizes(length=len(dataset), proportions=[.07,.07, .86])
    sets = random_split(dataset, sets_sizes, generator=Generator().manual_seed(c.seed)) # splits datasets randomly
    test_set, validation_set, train_set = sets


    # DATALOADER's
    train_loader = create_dataloader(train_set)
    validation_loader = create_dataloader(validation_set)
    test_loader = create_dataloader(test_set)

    # CREATING MODELS AND OPTIMIZERS
    sgvae = SGVAE().to(device)

    optimizer = torch.optim.Adam(sgvae.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, min_lr=min_lr, verbose=True
    )

    # LOSS FUNCTIONS
    reconstruction_loss = torch.nn.BCELoss()
    prediction_loss = torch.nn.L1Loss()

    # TODO: implement code to continue training
    # EPOCHS
    min_validation_loss = np.inf
    for epoch in range(0, epochs):
        print(f"epoch: {epoch+1}/{epochs}")
        start_time = time.perf_counter()
        
        # TRAINING
        sgvae.train()
        train_logs = {
            'elbo': 0, 'kl': 0, 'reconstruction_error': 0,
            f'mae_{properties[0]}': 0, f'mae_{properties[1]}': 0,
        }
        for x_input, x_output, label_0, label_1 in train_loader:
            optimizer.zero_grad()

            # TODO: remove data transfer after setup dataloader set device
            x_input = x_input.to(device)
            x_output = x_output.to(device)
            label_0 = label_0.to(device)
            label_1 = label_1.to(device)

            # TODO: convert model forwarding and loss calculations into a function
            # TODO: check if transpose operation could be removed
            x_input = x_input.transpose(1, 2).contiguous()  # [batch, num_rules, max_length]            
            z, mu, sigma, logits, predictions_0, predictions_1 = sgvae(x_input)
            x_decoded_mean = sgvae.conditional(x_output, logits)
            
            # ERRORS
            kl_divergence = sgvae.kl(mu, sigma)
            reconstruction_error = c.max_length * reconstruction_loss(x_decoded_mean.view(-1), x_output.view(-1)) 
            property_error_0 = prediction_loss(predictions_0.view(-1), label_0) 
            property_error_1 = prediction_loss(predictions_1.view(-1), label_1) 

            elbo =  (
                reconstruction_error * reconstruction_weight +
                kl_divergence * kl_weight +
                (property_error_0 + property_error_1) * prediction_weight
            )
            
            # UPDATE PARAMETERS
            elbo.backward()
            optimizer.step()
        
            # ADD ERROR PER BATCH
            train_logs['elbo'] += elbo.item()
            train_logs['kl'] += kl_divergence.item()
            train_logs['reconstruction_error'] += reconstruction_error.item()
            train_logs[f'mae_{properties[0]}'] += property_error_0.item()
            train_logs[f'mae_{properties[1]}'] += property_error_1.item()

        train_logs['elbo'] =                 train_logs['elbo'] / len(train_loader)
        train_logs['kl'] =                   train_logs['kl'] / len(train_loader)
        train_logs['reconstruction_error'] = train_logs['reconstruction_error'] / len(train_loader)
        train_logs[f'mae_{properties[0]}'] =      train_logs[f'mae_{properties[0]}'] / len(train_loader)
        train_logs[f'mae_{properties[1]}'] =      train_logs[f'mae_{properties[1]}'] / len(train_loader)

        writer.add_scalars('Loss/train', train_logs, global_step=epoch)
        train_registry.register('epoch', train_logs)
        # END TRAINING


        # VALIDATION
        validation_logs = {
            'elbo': 0, 'kl': 0, 'reconstruction_error': 0,
            f'mae_{properties[0]}': 0, f'mae_{properties[1]}': 0,
        }

        sgvae.eval()
        with torch.no_grad():
            for x_input, x_output, label_0, label_1 in validation_loader:
                # TODO: remove data transfer after setup dataloader to do this job
                x_input = x_input.to(device)
                x_output = x_output.to(device)
                label_0 = label_0.to(device)
                label_1 = label_1.to(device)

                # TODO: convert model forwarding and loss calculations into a function
                # TODO: check if transpose operation could be removed
                x_input = x_input.transpose(1, 2).contiguous()  # [batch, NUM_OF_RULES, MAX_LEN]
                z, mu, sigma, logits, predictions_0, predictions_1 = sgvae(x_input)
                x_decoded_mean = sgvae.conditional(x_output, logits)
                
                # ERRORS
                kl_divergence = sgvae.kl(mu, sigma)
                reconstruction_error = c.max_length * reconstruction_loss(x_decoded_mean.view(-1), x_output.view(-1)) 
                property_error_0 = prediction_loss(predictions_0.view(-1), label_0) 
                property_error_1 = prediction_loss(predictions_1.view(-1), label_1) 

                elbo =  (
                    reconstruction_error * reconstruction_weight +
                    kl_divergence * kl_weight +
                    (property_error_0 + property_error_1) * prediction_weight
                )
            
                # ADD ERROR PER BATCH
                validation_logs['elbo'] += elbo.item()
                validation_logs['kl'] += kl_divergence.item()
                validation_logs['reconstruction_error'] += reconstruction_error.item()
                validation_logs[f'mae_{properties[0]}'] += property_error_0.item()
                validation_logs[f'mae_{properties[1]}'] += property_error_1.item()

        # CaLCULATE LOGS MEAN'S
        validation_logs['elbo'] /= len(validation_loader)
        validation_logs['kl'] /= len(validation_loader)
        validation_logs['reconstruction_error'] /= len(validation_loader)
        validation_logs[f'mae_{properties[0]}'] /= len(validation_loader)
        validation_logs[f'mae_{properties[1]}'] /= len(validation_loader)

        validation_loss = validation_logs['elbo']

        writer.add_scalars('Loss/validation', validation_logs, global_step=epoch)
        validation_registry.register('epoch', validation_logs)
        # END VALIDATION

        # SAVING MODELS
        last_models_path = saving_path / 'last_models'
        sgvae.save(last_models_path)
        torch.save(optimizer.state_dict(), last_models_path / f'optimizer.pth')
        if (min_validation_loss > validation_loss):
            sgvae.save(saving_path / 'best_models')
            min_validation_loss = validation_loss
        
        scheduler.step(validation_loss)
        epoch_time = time.perf_counter() - start_time
        print(f'Time: {time.strftime("%H:%M:%S", time.gmtime(epoch_time))}')
        print(f'Train losses: {train_logs}')
        print(f'Validation losses: {validation_logs}\n')
    # END EPOCHS

    writer.close()

import parameters_parser
from datasets.processed_dataset import ProcessedDataset
from datasets.transforms import SGVAEDataFormatter, TwoSmilesFormatter
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

num_workers = math.ceil((multiprocessing.cpu_count() - 1)/2)

# TODO: update pytorch and use proportions directly in random_split
def calc_sets_splits_sizes(length, proportions: list):
    if(sum(proportions) != 1): raise ValueError('Sum of proportions is not 1 during split operation')


    splits_sizes = [math.floor(percentual*length) for percentual in proportions[:-1]]
    splits_sizes.append(length - sum(splits_sizes)) # add remains in the last split 

    return splits_sizes


"""
Creates a DataLoader which loads batches from dataset instead of a item
"""
def create_dataloader(dataset, batch_size):
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
    print('[TRAIN] Starting...')
    

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
        TwoSmilesFormatter(input_field='smiles', output_field='smiles', properties=properties)
    )


    # SPLIT SETS
    sets_sizes = calc_sets_splits_sizes(length=len(dataset), proportions=[.07,.07, .86])
    sets = random_split(dataset, sets_sizes, generator=Generator().manual_seed(c.seed)) # splits datasets randomly
    test_set, validation_set, train_set = sets

    # TODO: extract to class Split
    for data_set, name in zip(sets, ['test', 'validation', 'train']):
        with open(saving_path / f'{name}.indices', 'w') as file:
            file.write('\n'.join([str(i) for i in data_set.indices]))
    
    # DATALOADER's
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    assert len(train_loader)        >= 0, 'Dataset too short. Train loader has length 0.'
    assert len(validation_loader)   >= 0, 'Dataset too short. Validation loader has length 0.'
    assert len(test_loader)         >= 0, 'Dataset too short. Test loader has length 0.'

    # CREATING MODELS AND OPTIMIZERS
    sgvae = SGVAE(properties=properties).to(device)

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
        print(f"[EPOCH] {epoch+1}/{epochs}")
        start_time = time.perf_counter()
        
        # TRAINING
        sgvae.train()
        logs = { 'elbo': 0, 'kl': 0, 'reconstruction_error': 0 }
        for prop in properties: logs[f'mae_{prop}'] = 0

        for batch in train_loader:
            x_input, x_output, labels = batch['input'], batch['output'], batch['properties'],
            optimizer.zero_grad()

            # TODO: remove data transfer after setup dataloader set device
            x_input = x_input.to(device, dtype=torch.float)
            x_output = x_output.to(device, dtype=torch.float)
            labels = { prop: label.to(device, dtype=torch.float) for prop, label in labels.items()}

            # TODO: convert model forwarding and loss calculations into a function
            # TODO: check if transpose operation could be removed
            x_input = x_input.transpose(1, 2).contiguous()  # [batch, num_rules, max_length]
            z, mu, sigma, logits, predictions = sgvae(x_input)
            x_decoded_mean = sgvae.conditional(x_output, logits)
            
            # ERRORS
            kl_divergence = sgvae.kl(mu, sigma)
            reconstruction_error = c.max_length * reconstruction_loss(x_decoded_mean.view(-1), x_output.view(-1)) 
            predictions_errors = {
                prop: prediction_loss(predictions[prop].view(-1), labels[prop]) for prop in properties
            }
            sum_predicitions_errors = sum(predictions_errors.values())

            elbo =  (
                reconstruction_error * reconstruction_weight +
                kl_divergence * kl_weight +
                sum_predicitions_errors * prediction_weight
            )
        
            # ADD ERROR PER BATCH
            logs['elbo'] += elbo.item()
            logs['kl'] += kl_divergence.item()
            logs['reconstruction_error'] += reconstruction_error.item()
            for prop in properties: logs[f'mae_{prop}'] += predictions_errors[prop].item()
            
            # UPDATE PARAMETERS
            elbo.backward()
            optimizer.step()

        logs['elbo'] /= len(train_loader)
        logs['kl'] /= len(train_loader)
        logs['reconstruction_error'] /= len(train_loader)
        for prop in properties: logs[f'mae_{prop}'] /= len(train_loader)

        writer.add_scalars('Loss/train', logs, global_step=epoch)
        train_registry.register('epoch', logs)
        train_logs = logs
        # END TRAINING


        # VALIDATION
        sgvae.eval()

        logs = { 'elbo': 0, 'kl': 0, 'reconstruction_error': 0 }
        for prop in properties: logs[f'mae_{prop}'] = 0

        with torch.no_grad():
            for batch in validation_loader:
                x_input, x_output, labels = batch['input'], batch['output'], batch['properties'],
                
                # TODO: remove data transfer after setup dataloader set device
                x_input = x_input.to(device, dtype=torch.float)
                x_output = x_output.to(device, dtype=torch.float)
                labels = { prop: label.to(device, dtype=torch.float) for prop, label in labels.items()}


                # TODO: convert model forwarding and loss calculations into a function
                # TODO: check if transpose operation could be removed
                x_input = x_input.transpose(1, 2).contiguous()  # [batch, NUM_OF_RULES, MAX_LEN]
                z, mu, sigma, logits, predictions = sgvae(x_input)
                x_decoded_mean = sgvae.conditional(x_output, logits)
                
                # ERRORS
                kl_divergence = sgvae.kl(mu, sigma)
                reconstruction_error = c.max_length * reconstruction_loss(x_decoded_mean.view(-1), x_output.view(-1)) 
                predictions_errors = {
                    prop: prediction_loss(predictions[prop].view(-1), labels[prop]) for prop in properties
                }
                sum_predicitions_errors = sum(predictions_errors.values())
                
                elbo =  (
                    reconstruction_error * reconstruction_weight +
                    kl_divergence * kl_weight +
                    sum_predicitions_errors * prediction_weight
                )
            
                # ADD ERROR PER BATCH
                logs['elbo'] += elbo.item()
                logs['kl'] += kl_divergence.item()
                logs['reconstruction_error'] += reconstruction_error.item()
                for prop in properties: logs[f'mae_{prop}'] += predictions_errors[prop].item()
            
        # CaLCULATE LOGS MEAN'S
        logs['elbo'] /= len(train_loader)
        logs['kl'] /= len(train_loader)
        logs['reconstruction_error'] /= len(train_loader)
        for prop in properties: logs[f'mae_{prop}'] /= len(train_loader)


        validation_loss = logs['elbo']

        writer.add_scalars('Loss/validation', logs, global_step=epoch)
        validation_registry.register('epoch', logs)
        validation_logs = logs
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
        print(f'\t[INFO] Time: {time.strftime("%H:%M:%S", time.gmtime(epoch_time))}')
        print(f'\t[INFO] Train losses: {train_logs}')
        print(f'\t[INFO] Validation losses: {validation_logs}')
    # END EPOCHS

    writer.close()

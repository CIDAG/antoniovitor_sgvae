#!/usr/bin/env python3
import click
import bootstrap
from scripts import process_dataset
from scripts import sample

@click.group()
def cli(): pass

# TODO: declare command in script's files. Currently, the train.run import is
# too low to be executed in every call.
# TODO: add arguments dataset, subset, properties
@cli.command(name='train')
@click.option(
    '--dataset_name',
    type=click.STRING,
    prompt='Dataset name',
    required=True,
)
@click.option(
    '--batch_size',
    type=click.INT,
    default=512,
)
def train_command(**kargs):
    from scripts import train
    train.run(**kargs)


def main():
    bootstrap.run()
    cli.add_command(process_dataset.run, name='process_dataset')
    cli.add_command(sample.run, name='sample')
    cli()

if __name__ == '__main__':
    main()

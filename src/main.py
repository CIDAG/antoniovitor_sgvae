#!/usr/bin/env python3
import click
from scripts import preprocess_dataset

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
def train_command(**kargs):
    from scripts import train
    train.run(**kargs)


def main():
    cli.add_command(preprocess_dataset.run, name='preprocess_dataset')
    cli()

if __name__ == '__main__':
    main()

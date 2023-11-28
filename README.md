# Commands

All commands and parameters are documented automatically. If you need any help,
run the command:

```ssh
python main.py --help 
# or
python main.py process_dataset --help 
# or
python main.py train --help 
```

## Processing datasets

Any CSV file could be used as a dataset to train SGVAE. To format a new dataset
to be used as training data, it is possible to execute the following command to
process the data.

```ssh
python main.py process_dataset \
    --path=../datasources/IL_ESW/dataset_complete/anions.csv \
    --dataset_name=il_esw_anions \
    -p homo -p lumo -p viscosity 
```

- `--path`: is the path to the CSV file
- `--dataset_name`: is the name used to save the dataset. This will be used to reference the dataset in other scripts.
- `--properties_fields`: are the names of the two properties that will be used as target value to the model.
- `--ion_type` (optional): type of the ion (anion or cation) presented in the dataset
- `--num_samples` (optional): number of samples that should be selected. This is the desired final length of the dataset, but other steps of the preprocessing procedure could make the final length smaller than num_samples.

## Training 

To train a new version, run the following command:

```ssh
python main.py train --dataset_name=il_esw_anions
```

- `--dataset_name`: is the name given to the dataset.
- `--batch_size` (optional):  is the size of the batch used during training.
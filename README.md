# Commands

All commands and parameters are documented automatically. If you need any help,
run the command:

```ssh
python main.py --help 
# or
python main.py preprocess_dataset --help 
# or
python main.py train --help 
```

## Processing datasets

Any CSV file could be used as a dataset to train SGVAE. To format a new dataset
to be used as training data, it is possible to execute the following command to
process the data.

```ssh
python main.py preprocess_dataset --path=../datasources/IL_ESW/dataset_complete/anions.csv --dataset_name=il_esw_anions --properties_fields homo-fopt lumo-fopt 
```

- `--path` is the path to the CSV file
- `--dataset_name` is the name used to save the dataset. This will be used to reference the dataset in other scripts.
- `--properties_fields` are the names of the two properties that will be used as target value to the model.

## Training 

To train a new version, run the following command:

```ssh
python main.py train --dataset_name=il_esw_anions
```

- `--dataset_name` is the name given to the dataset.
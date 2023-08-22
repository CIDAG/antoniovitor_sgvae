import nltk
import pdb
import grammar
import numpy as np
import h5py
import pickle
import random
import grammar
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from datasets.IL_ESW_extended import ESW_IL_Extended_Subset
random.seed(42)


# df = pd.read_csv('../datasources/IL_ESW_extended/anions.csv')
# list_df = list(df.index)

# # choosing random ids for train and test
# ids = list(range(len(list_df)))
# random.shuffle(ids)

# chunk = int(0.15 * len(list_df))
# ids_train = sorted(ids[chunk:])
# ids_test = sorted(ids[0:chunk])

# L = [list_df[i] for i in ids_train] # smiles for training
# L_test = [list_df[i] for i in ids_test] # smiles for test

# smiles_list = df['smiles'].to_list()

dataset = ESW_IL_Extended_Subset(subset='anions')
smiles_list = [i['smiles'] for i in dataset]
# L_property = [list_prop[i] for i in ids_train] # smiles for training
# L_property_test = [list_prop[i] for i in ids_test] # smiles for test

MAX_LEN = 492
NCHARS = len(grammar.utils.GCFG.productions())
len_seq = []

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(grammar.utils.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = grammar.utils.get_zinc_tokenizer(grammar.utils.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(grammar.utils.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    for seq in productions_seq:
         len_seq.append(len(seq))


def main():
    
    for i in tqdm(range(0, len(smiles_list), 1000)):
        to_one_hot(smiles_list[i:i+1000])

    print(f'max: {max(len_seq)}')     
  #  h5f = h5py.File('data/anions_grammar_dataset.h5','w')
  #  h5f.create_dataset('data', data=OH)
  #  h5f.close()
    
if __name__ == '__main__':
    main()


from . import utils
import numpy as np
import nltk
import parameters_parser
from tqdm import tqdm
import constants as c

# TODO: remove D from module (the dimension should be passed as a parameter)
D = utils.D
# TODO: optimize generation of grammar variables
masks = utils.masks
ind_of_ind = utils.ind_of_ind


def parse_smiles(smiles, MAX_LEN, NCHARS):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(utils.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = utils.get_zinc_tokenizer(utils.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(utils.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


def parse_smiles_list(smiles_list, verbose=False):
    params = parameters_parser.load_parameters()
    MAX_LEN = c.max_length
    NCHARS = len(utils.GCFG.productions())

    OH = np.zeros((len(smiles_list),MAX_LEN,NCHARS))
    for i in tqdm(range(0, len(smiles_list), 100), disable=not verbose):
        onehot = parse_smiles(smiles_list[i:i+100], MAX_LEN, NCHARS)
        OH[i:i+100,:,:] = onehot

    return OH
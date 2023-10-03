import nltk
import numpy as np
import six
import pdb
import torch
import grammar.utils as utils
import constants as c
from tqdm import tqdm

# TODO: remove old_grammar from project

device = "cuda" if torch.cuda.is_available() else "cpu"

gram = """smiles -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'B'
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'S'
aliphatic_organic -> 'P'
aliphatic_organic -> 'F'
aliphatic_organic -> 'I'
aliphatic_organic -> 'Cl'
aliphatic_organic -> 'Br'
aliphatic_organic -> 'Si'
aliphatic_organic -> 'Al'
aliphatic_organic -> 'Ga'
aliphatic_organic -> 'Sb'
aliphatic_organic -> 'In'
aliphatic_organic -> 'Nb'
aliphatic_organic -> 'Ta'
aliphatic_organic -> 'As'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> BAH
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
Nothing -> None"""

# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

# this map tells us the rhs symbol indices for each production rule
rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list),D))
count = 0

# this tells us for each lhs symbol which productions rules should be masked
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

# this tells us the indices where the masks are equal to 1
index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:,i]==1)[0][0])
ind_of_ind = np.array(index_array)
ind_of_ind = torch.Tensor(ind_of_ind).to(device)

max_rhs = max([len(l) for l in rhs_map])
masks = torch.ByteTensor(masks).to(device)


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
    MAX_LEN = c.max_length
    NCHARS = len(utils.GCFG.productions())

    OH = np.zeros((len(smiles_list),MAX_LEN,NCHARS))
    for i in tqdm(range(0, len(smiles_list), 100), disable=not verbose):
        onehot = parse_smiles(smiles_list[i:i+100], MAX_LEN, NCHARS)
        OH[i:i+100,:,:] = onehot

    return OH
import nltk
import six
from .rules import rules
import numpy as np
import constants as c
from tqdm import tqdm

class Grammar():
    _cfg: nltk.CFG = None
    _all_lhs = None
    _lhs_list = None
    _rhs_map = None
    _masks = None
    _ind_of_ind = None
    
    @property
    def cfg(self) -> nltk.CFG:
        if self._cfg is None:
            self._cfg = nltk.CFG.fromstring(rules)

        return self._cfg
    
    @property
    def productions(self):
        return self.cfg.productions()

    @property
    def rules_length(self):
        return len(self.productions)

    @property
    def start_index(self):
        # TODO: change to self.cfg.start() if there is no dependency
        # return self.cfg.start()
        return self.productions[0].lhs()

    @property
    def all_lhs(self):
        if(self._all_lhs is None):
            self._all_lhs = [a.lhs().symbol() for a in self.productions]
        return self._all_lhs

    @property
    def lhs_list(self):
        # TODO: change lhs_list to set
        # return set(a.lhs().symbol() for a in self.productions)
        
        if(self._lhs_list is None):
            lhs_list = []
            for a in self.all_lhs:
                if a not in lhs_list:
                    lhs_list.append(a)
            self._lhs_list = lhs_list

        return self._lhs_list

    # this map tells us the rhs symbol indices for each production rule
    @property
    def rhs_map(self):
        if(self._rhs_map is None):
            rhs_map = [None]*self.rules_length
            count = 0
            for a in self.productions:
                rhs_map[count] = []
                for b in a.rhs():
                    if not isinstance(b,six.string_types):
                        s = b.symbol()
                        rhs_map[count].extend(list(np.where(np.array(self.lhs_list) == s)[0]))
                count = count + 1
            self._rhs_map = rhs_map

        return self._rhs_map

    @property
    def masks(self):
        if(self._masks is None):
            masks = np.zeros((len(self.lhs_list), self.rules_length))
            count = 0

            # this tells us for each lhs symbol which productions rules should be masked
            for sym in self.lhs_list:
                is_in = np.array([a == sym for a in self.all_lhs], dtype=int).reshape(1,-1)
                masks[count] = is_in
                count = count + 1
            self._masks = masks
        
        return self._masks

    @property
    def ind_of_ind(self):
        # this tells us the indices where the masks are equal to 1
        if(self._ind_of_ind is None):
            index_array = []
            for i in range(self.masks.shape[1]):
                index_array.append(np.where(self.masks[:,i]==1)[0][0])
            self._ind_of_ind = np.array(index_array)

        return self._ind_of_ind

    def parse_smiles(self, smiles, MAX_LEN, NCHARS):
        """ Encode a list of smiles strings to one-hot vectors """
        assert type(smiles) == list
        prod_map = {}
        for ix, prod in enumerate(self.productions):
            prod_map[prod] = ix
        tokenize = self.get_zinc_tokenizer()
        tokens = map(tokenize, smiles)
        parser = nltk.ChartParser(self.cfg)
        parse_trees = [parser.parse(t).__next__() for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
        return one_hot


    def parse_smiles_list(self, smiles_list, verbose=False):
        MAX_LEN = c.max_length
        NCHARS = len(self.productions)

        OH = np.zeros((len(smiles_list),MAX_LEN,NCHARS))
        for i in tqdm(range(0, len(smiles_list), 100), disable=not verbose):
            onehot = self.parse_smiles(smiles_list[i:i+100], MAX_LEN, NCHARS)
            OH[i:i+100,:,:] = onehot

        return OH
    
    def get_zinc_tokenizer(self):
        long_tokens = [a for a in list(self.cfg._lexical_index.keys()) if len(a) > 1]
        replacements = ['$','%','^','~','&','!','?','°','>','<', '|']
        try:
            assert len(long_tokens) == len(replacements)
        except AssertionError:
            print(f'\nDifferent lenghts encountered. The lenght of the token replacement is {len(replacements)}, but the lenght of the long tokens in {len(long_tokens)}.')
        
        for token in replacements: 
            assert token not in self.cfg._lexical_index
        
        def tokenize(smiles):
            for i, token in enumerate(long_tokens):
                smiles = smiles.replace(token, replacements[i])
            tokens = []
            for token in smiles:
                try:
                    ix = replacements.index(token)
                    tokens.append(long_tokens[ix])
                except:
                    tokens.append(token)
            return tokens
        
        return tokenize


    def pop_or_nothing(self, S):
        try: return S.pop()
        except: return 'Nothing'


    def prods_to_eq(self, prods):
        seq = [prods[0].lhs()]
        for prod in prods:
            if str(prod.lhs()) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod.lhs():
                    seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                    break
        try:
            return ''.join(seq)
        except:
            return ''
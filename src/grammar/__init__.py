import nltk
import six
from . import rules
import numpy as np

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

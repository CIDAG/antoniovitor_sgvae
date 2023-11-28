import torch
from .encoder import Encoder
from .decoder import Decoder
from .property_predictor import PropertyPredictor
import constants as c
import grammar.old_grammar as grammar
# from grammar import Grammar
import parameters_parser
from pathlib import Path
import numpy as np
import nltk

# TODO: remove the to(device) which requires parameters
params = parameters_parser.load_parameters()
device = params['device']

class SGVAE(torch.nn.Module):
    def __init__(self, properties) -> None:
        super(SGVAE, self).__init__()

        # TODO: remove old grammar
        self._grammar = grammar
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = grammar.utils.get_zinc_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix

        self.input_shape = (c.num_rules, c.max_length)
        self.encoder = Encoder(input_shape=self.input_shape, output_size=c.latent_size)
        self.decoder = Decoder(input_size=c.latent_size,  output_size=c.num_rules,
                               max_length=c.max_length)
        
        self.properties = properties
        self.predictors = torch.nn.ModuleDict({
            prop: PropertyPredictor(input_size=c.latent_size) for prop in self.properties
        })
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        logits = self.decoder(z)

        predictions = {
            prop: self.predictors[prop](z) for prop in self.properties
        }
        
        # returning x to its original dimensions
        return z, mu, sigma, logits, predictions

    def load(self, models_path):
        self.encoder.load_state_dict(torch.load(models_path / 'encoder.pth'))
        self.decoder.load_state_dict(torch.load(models_path / 'decoder.pth'))
        for prop in self.properties:
            self.predictors[prop].load_state_dict(torch.load(models_path / f'predictor_{prop}.pth'))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(self.encoder.state_dict(), folder / 'encoder.pth')
        torch.save(self.decoder.state_dict(), folder / 'decoder.pth')
        for prop in self.properties:
            torch.save(self.predictors[prop].state_dict(), folder / f'predictor_{prop}.pth')

    def sample(self, mu, sigma):
        """Reparametrization trick to sample z"""
        sigma = torch.exp(0.5 * sigma)        
        epsilon = torch.randn(len(mu), c.latent_size).to(device)
        
        return mu + sigma * epsilon  

    def kl(self, mu, sigma):
        """KL divergence between the approximated posterior and the prior"""
        return - 0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp()) 

    def conditional(self, x_true, x_pred):
        most_likely = torch.argmax(x_true, dim=-1)
        most_likely = most_likely.view(-1) # flatten most_likely
        ix2 = torch.unsqueeze(grammar.ind_of_ind[most_likely], -1) # index ind_of_ind with res
        ix2 = ix2.type(torch.LongTensor)
        M2 = grammar.masks[list(ix2.T)]
        # M3 = torch.reshape(M2, (params['batch'], params['max_length'], grammar.D))
        M3 = torch.reshape(M2, (len(x_true), c.max_length, c.num_rules))
        P2 = torch.mul(torch.exp(x_pred), M3.float()) # apply them to the exp-predictions
        P2 = torch.divide(P2, torch.sum(P2, dim=-1, keepdims=True)) # normalize predictions
        return P2

    def encode(self, smiles):
        """ 
        Encode a list of smiles strings into the latent space. The input smiles
        is a list of regular smiles which were not used for training. 
        """
        assert type(smiles) == list
        tokens = map(self._tokenize, smiles)
        parse_trees = [self._parser.parse(t).__next__() for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), c.max_length, self._n_chars), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, c.max_length),-1] = 1.
        one_hot = torch.from_numpy(one_hot).to(device)  # [batch, MAX_LEN, NUM_OF_RULES]
        one_hot = one_hot.transpose(1, 2)  # need to reshape to [batch, NUM_OF_RULES, MAX_LEN] for the convolution encoder

        return self.encoder(one_hot)[0]

    def _sample_using_masks(self, unmasked):
        """ 
        Samples a one-hot vector, masking at each timestep. This is an 
        implementation of Algorithm 1 in the paper. Notice that unmasked is a
        torch tensor
        """
        eps = 1e-10
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self._grammar.start_index)]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self._lhs_map[grammar.utils.pop_or_nothing(a)] for a in S]
            mask = self._grammar.masks[next_nonterminal]
            masked_output = np.exp(unmasked[:,t,:]) * mask.cpu().detach().numpy() + eps #.cpu().detach().numpy()
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                        self._productions[i].rhs()) 
                    for i in sampled_output]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
        return X_hat


    def decode(self, z):
        """ Sample from the grammar decoder """

        unmasked = self.decoder(z)
        unmasked = unmasked.cpu().detach().numpy()

        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[X_hat[index,t].argmax()] 
                    for t in range(X_hat.shape[1])] 
                    for index in range(X_hat.shape[0])]
                        
        return [grammar.utils.prods_to_eq(prods) for prods in prod_seq]
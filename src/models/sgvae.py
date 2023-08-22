import torch
from .encoder import Encoder
from .decoder import Decoder
from .property_predictor import PropertyPredictor
import constants as c
import grammar
import parameters_parser
from pathlib import Path

# TODO: remove the to(device) which requires parameters
params = parameters_parser.load_parameters()
device = params['device']

class SGVAE(torch.nn.Module):
    def __init__(self) -> None:
        super(SGVAE, self).__init__()
        self.input_shape = (c.num_rules, c.max_length)
        self.encoder = Encoder(input_shape=self.input_shape, output_size=c.latent_size)
        self.decoder = Decoder(input_size=c.latent_size,  output_size=c.num_rules,
                               max_length=c.max_length)
        self.predictor_0 = PropertyPredictor(input_size=c.latent_size)
        self.predictor_1 = PropertyPredictor(input_size=c.latent_size)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        logits = self.decoder(z)

        predictions_0 = self.predictor_0(z)
        predictions_1 = self.predictor_1(z)
        
        # returning x to its original dimensions
        return z, mu, sigma, logits, predictions_0, predictions_1

    def load(self, models_path):
        self.encoder.load_state_dict(torch.load(models_path / 'encoder.pth'))
        self.decoder.load_state_dict(torch.load(models_path / 'decoder.pth'))
        self.predictor_0.load_state_dict(torch.load(models_path / 'predictor_0.pth'))
        self.predictor_1.load_state_dict(torch.load(models_path / 'predictor_1.pth'))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(self.encoder.state_dict(), folder / 'encoder.pth')
        torch.save(self.decoder.state_dict(), folder / 'decoder.pth')
        torch.save(self.predictor_0.state_dict(), folder / 'predictor_0.pth')
        torch.save(self.predictor_1.state_dict(), folder / 'predictor_1.pth')

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
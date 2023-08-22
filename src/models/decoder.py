import torch.nn as nn

# TODO: remove max_length

class Decoder(nn.Module):
  """
  GRU decoder for the Grammar VAE

  The implementation is equivalent than the original paper, 
  only translated to pytorch
  """
  def __init__(self, input_size, output_size, max_length):
    super(Decoder, self).__init__()
    self.max_length = max_length

    self.linear_in = nn.Linear(input_size, input_size)
    self.rnn = nn.GRU(input_size = input_size, hidden_size = 501, num_layers = 3, batch_first=True)
    self.linear_out = nn.Linear(501, output_size)

    self.relu = nn.LeakyReLU()

  def forward(self, z):
    h = self.relu(self.linear_in(z))
    h = h.unsqueeze(1).expand(-1, self.max_length, -1)  #[batch, MAX_LENGHT, latent_dim] This does the same as the repeatvector on keras
    h, _ = self.rnn(h)
    h = self.relu(self.linear_out(h))

    return h
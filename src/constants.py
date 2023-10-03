import torch

# GENERAL
seed = 42

# MODELS CONSTANT
num_rules = 71 # from grammar.D (number of rules)
max_length = 494
latent_size = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
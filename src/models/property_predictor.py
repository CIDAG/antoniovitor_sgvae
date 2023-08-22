import torch
import torch.nn as nn

hidden_size = 70

class PropertyPredictor(nn.Module):
    def __init__(self, input_size) -> None:
        super(PropertyPredictor, self).__init__()
        
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, z):
        return self.feedforward(z)
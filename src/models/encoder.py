import torch
from torch import nn

class Encoder(nn.Module):
    """
    Convolutional encoder for the Grammar VAE
    The implementation is equivalent to the original paper, 
    only translated to pytorch
    """
    def __init__(self, input_shape, output_size):
        super(Encoder, self).__init__()        
        self.input_size, self.max_length = input_shape
        # convolution 1
        self.out_channels_1 = 9
        self.kernel_1 = 9
        self.conv1 = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.out_channels_1,
            kernel_size=self.kernel_1
        )
        # convolution 2
        self.out_channels_2 = 9
        self.kernel_2 = 9
        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels_1,
            out_channels=self.out_channels_2,
            kernel_size=self.kernel_2
        )
        # convolution 3
        self.out_channels_3 = 10
        self.kernel_3 = 11
        self.conv3 = nn.Conv1d(
            in_channels=self.out_channels_2,
            out_channels=self.out_channels_3,
            kernel_size=self.kernel_3
        )

        self.linear_size = (
            self.max_length
            - (self.kernel_1 - 1)
            - (self.kernel_2 - 1)
            - (self.kernel_3 - 1)
        ) * self.out_channels_3

        self.linear = nn.Linear(self.linear_size, 435)

        self.mu = nn.Linear(435, output_size)
        self.sigma = nn.Linear(435, output_size)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = torch.transpose(h, 1, 2)  # need to transpose to get the right output
        h = h.contiguous().view(h.size(0), -1) # flatte
        h = self.relu(self.linear(h))
        
        mu = self.mu(h)
        sigma = self.sigma(h)

        return mu, sigma
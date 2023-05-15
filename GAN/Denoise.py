import torch
import torch.nn as nn
import numpy as np

device = torch.device('mps')

# Define the GAN model for denoising
class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 100)
        )
        # Define the discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward propagate the encoder
        out = self.encoder(x)
        # Forward propagate the decoder
        out = self.decoder(out)
        return out

    def discriminator(self, x):
        # Forward propagate the discriminator
        out = self.discriminator(x)
        return out
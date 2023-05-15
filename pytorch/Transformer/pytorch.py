import torch
import torch.nn as nn
import numpy as np
from random import randint
from matplotlib import pyplot as plt

# Device configuration
device = torch.device('cpu')

# Define the embedding layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x be a shape of (batch_size, max_seq_len, d_model)
        x = x + self.pe[:x.size(1), :]
        return x

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # Define the embedding layer
        self.embedding = nn.Embedding(100, 128)
        # Define the positional encoding layer
        self.positional_encoding = PositionalEncoding(128, 100)
        # Define the transformer layer
        self.transformer = nn.Transformer(128, 3, 3, 3, 3, 0.1)
        # Define the output layer
        self.fc = nn.Linear(128, 100)

    def forward(self, x):
        # Forward propagate the embedding layer
        out = self.embedding(x)
        # Forward propagate the positional encoding layer
        out = self.positional_encoding(out)
        # Forward propagate the transformer layer
        out = self.transformer(out, out)
        # Forward propagate the output layer
        out = self.fc(out)
        return out


model = Transformer().to(device)

# Get the Training Data

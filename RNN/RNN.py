import torch
import torch.nn as nn
import numpy as np
from random import randint
from matplotlib import pyplot as plt

# Device configuration
device = torch.device('cpu')

sample_size = 1000
data = torch.Tensor()
output = torch.Tensor()
for i in range(sample_size):
    # x be a shape of (1, 100) tensor
    x = np.random.uniform(-10, 10, 100)
    y = randint(-10, 10) * x ** 2 + randint(-10, 10) * x + randint(-10, 10)
    noise = np.random.normal(0, 1, 100)
    y_noise = y + noise
    # make sure the dtype is float32
    y = y.astype(np.float32)
    y_noise = y_noise.astype(np.float32)
    data = torch.cat((data, torch.from_numpy(y_noise).view(1, -1)), 0)
    output = torch.cat((output, torch.from_numpy(y).view(1, -1)), 0)

data = data.view(sample_size, 1, -1)
output = output.view(sample_size, 1, -1)


# Hyper-parameters

# Define the model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # Define the RNN layer
        self.rnn = nn.LSTM(100, 128, 3, batch_first=True)
        # Define the output layer
        self.fc = nn.Linear(128, 100)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(3, x.size(0), 128).to(device)
        c0 = torch.zeros(3, x.size(0), 128).to(device)
        # make sure the size of h0 and c0 is (num_layers, batch_size, hidden_size)
        # Forward propagate LSTM
        out, _ = self.rnn(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = RNN().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

losses = []
# Train the model
for epoch in range(2):
    for i in range(sample_size):
        # Forward pass
        outputs = model(data[i])
        loss = criterion(outputs, output[i])
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, 2, i + 1, sample_size, loss.item()))
    plt.plot(losses)
    plt.show()
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

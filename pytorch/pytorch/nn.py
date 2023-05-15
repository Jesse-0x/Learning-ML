import torch
import numpy as np
import math

# define a neural network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # inherit the init function of torch.nn.Module
        super(Net, self).__init__()
        # define the structure of neural network
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # forward propagation
        x = torch.sigmoid(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# log(x^2 + 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

print(x.shape)
print(y.shape)

# define the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.09)
# define the loss function
loss_func = torch.nn.MSELoss()

losses = []
for t in range(30000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    losses.append(loss.data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import matplotlib.pyplot as plt
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.show()

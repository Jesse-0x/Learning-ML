import numpy
import torch
import matplotlib.pyplot as plt
import random

device = torch.device('cpu')


# Define the GAN model for data denoising
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 128)
        )
        # Define the decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        # Forward propagate the encoder
        out = self.encoder(x)
        # Forward propagate the decoder
        out = self.decoder(out)
        return out


# Define the discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define the discriminator
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Forward propagate the discriminator
        out = self.discriminator(x)
        return out


# Define the loss function
def loss_function(x, x_hat, D_x_hat, D_x):
    # Compute the loss for the generator
    loss_g = torch.mean((x - x_hat) ** 2) + torch.mean((D_x_hat - 1) ** 2)
    # Compute the loss for the discriminator
    loss_d = torch.mean((D_x - 1) ** 2) + torch.mean(D_x_hat ** 2)
    return loss_g, loss_d


# Define the training function
def train(model, optimizer, x, x_hat, D_x_hat, D_x):
    # Compute the loss
    loss_g, loss_d = loss_function(x, x_hat, D_x_hat, D_x)
    # Clear the gradient
    optimizer.zero_grad()
    # Backward propagate the loss
    loss_g.backward(retain_graph=True)
    loss_d.backward()
    # Update the parameters
    optimizer.step()
    return loss_g, loss_d


# Define the training data generator
def data_generator(batch_size=100):
    # Generate the data
    x = numpy.random.normal(0, 1, (batch_size, 100))
    # Convert the data to a tensor
    x = torch.from_numpy(x).float()
    # add noise to the data
    x_noisy = x.clone()
    for i in range(batch_size):
        x_noisy[i] = x_noisy[i] + numpy.random.normal(0, 0.5, (1, 100))
    assert int((x-x_noisy).max()) != 0
    return x.to(device), x_noisy.to(device)

def graph(x, x_hat):
    # Convert the data to a numpy array
    x = x.detach().numpy()
    x_hat = x_hat.detach().numpy()
    # Plot the data
    plt.plot(x[0], label='x')
    plt.plot(x_hat[0], label='x_hat')
    plt.legend()
    plt.show()

def graph_as_picture(x, x_noise, x_hat):
    # Convert the data to a numpy array
    x = x.detach().numpy()
    x_noise = x_noise.detach().numpy()
    x_hat = x_hat.detach().numpy()
    # Plot the data as a gray picture
    plt.subplot(1, 3, 1)
    plt.imshow(x.reshape(10, 10), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(x_noise.reshape(10, 10), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(x_hat.reshape(10, 10), cmap='gray')
    plt.show()

if __name__ == '__main__':
    # Set the random seed
    torch.manual_seed(0)
    epochs = 3000
    batch_size = 1000
    learning_rate = 0.0001
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    # load from the checkpoint
    generator.load_state_dict(torch.load('generator.ckpt'))
    discriminator.load_state_dict(torch.load('discriminator.ckpt'))
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    # Train the model
    loss_gs = []
    loss_ds = []
    for epoch in range(epochs):
        # Generate the data
        x, x_noisy = data_generator(batch_size)
        # Forward propagate the generator
        x_hat = generator(x_noisy).to(device)
        # Forward propagate the discriminator
        D_x_hat = discriminator(x_hat).to(device)
        D_x = discriminator(x).to(device)
        # Train the model
        loss_g, loss_d = train(generator, optimizer, x, x_hat, D_x_hat, D_x)
        # Append the loss
        loss_gs.append(loss_g.item())
        loss_ds.append(loss_d.item())
        # Print the loss
        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss_g: {:.4f}, Loss_d: {:.4f}'.format(epoch + 1, epochs, loss_g.item(), loss_d.item()))
    # Generate the data
    x, x_noisy = data_generator(batch_size)
    # Forward propagate the generator
    x_hat = generator(x_noisy)
    # Plot the data
    # Plot the loss
    plt.plot(loss_gs, label='loss_g')
    plt.plot(loss_ds, label='loss_d')
    plt.legend()
    plt.show()

    # graph_as_picture(x[0], x_noisy[0], x_hat[0])
    # graph_as_picture(x[1], x_noisy[1], x_hat[1])
    # graph_as_picture(x[2], x_noisy[2], x_hat[2])

    # Save the model checkpoints
    torch.save(generator.state_dict(), 'generator.ckpt')
    torch.save(discriminator.state_dict(), 'discriminator.ckpt')

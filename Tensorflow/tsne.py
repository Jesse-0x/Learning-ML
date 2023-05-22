import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# Define the t-SNE model
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=5000, verbose=1)
# Fit the model to the data
x_tsne = tsne.fit_transform(x_train[:1000])

# Visualize the results
fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], c=y_train[:1000], cmap=plt.cm.get_cmap('jet', 10))
# plt.show()
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_train[:1000], cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

# apply
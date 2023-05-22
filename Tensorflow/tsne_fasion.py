import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the MNIST fashion dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# Define the t-SNE model
tsne = TSNE(n_components=3, perplexity=40, n_iter=5000, verbose=1)

# Fit the model to the data
x_tsne = tsne.fit_transform(x_train[:1000])

# Visualize the results
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], c=y_train[:1000], cmap=plt.cm.get_cmap('jet', 10))
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = FuncAnimation(fig, lambda i: ax.view_init(elev=10., azim=i), frames=360, interval=20)
anim.save('fashion_mnist_tsne.mp4', writer=writer)
# LDA for minst

import tensorflow as tf
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# Define the LDA model
lda = LinearDiscriminantAnalysis(n_components=3)
# Fit the model to the data
x_lda = lda.fit_transform(x_train, y_train)

# Visualize the results with 3D
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(x_lda[:, 0], x_lda[:, 1], x_lda[:, 2], c=y_train, cmap=plt.cm.get_cmap('jet', 10))
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = FuncAnimation(fig, lambda i: ax.view_init(elev=10., azim=i), frames=360, interval=20)
anim.save('mnist_lda.mp4', writer=writer)



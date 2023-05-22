import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load some sine and cosine data
x1 = np.linspace(0, 2*np.pi, 100)
x2 = np.linspace(0, 2*np.pi, 100)
x = np.concatenate((x1, x2))
y1 = np.sin(x1)
y2 = np.cos(x2)
y = np.concatenate((y1, y2))

# Define the t-SNE model
# tsne = TSNE(n_components=2, perplexity=40.0, n_iter=5000, verbose=1)
# /Fit the model to the data
# x_tsne = tsne.fit_transform(np.vstack((x, y)).T)
# decide to
# PCA
# Visualize the results
fig = plt.figure()
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 2))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

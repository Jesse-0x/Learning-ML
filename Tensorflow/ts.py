import tensorflow as tf


def log_t(u, t):
    """Compute log_t for `u`."""
    return (u ** (1 - t) - 1) / (1 - t) if t != 1 else tf.math.log(u)


def exp_t(u, t):
    """Compute exp_t for `u`."""
    return tf.nn.relu(1 + (1 - t) * tf.math.log(u)) ** (1 / (1 - t)) if t != 1 else tf.math.exp(u)


def compute_normalization(z, t, num_iters=5):
    """Compute the normalization term `Z` for a given temperature `t`."""
    # We use a fixed-point iteration to compute Z
    mu = tf.reduce_max(z, axis=-1, keepdims=True)
    normalized_z = z - mu
    for i in range(num_iters):
        log_z = tf.reduce_sum(exp_t(normalized_z, t), axis=-1, keepdims=True)
        normalized_z = z - mu - log_t(log_z, t)
    log_z = tf.reduce_sum(exp_t(normalized_z, t), axis=-1, keepdims=True)
    return mu + log_t(log_z, t)


def compute_bi_tempered_logistic_loss(y_pred, y_true, t1=0.8, t2=1.4, label_smoothing=0.1):
    """Compute the Robust Bi-Tempered Logistic Loss."""
    num_classes = y_pred.shape[-1]
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    temp1 = (1 - y_true) * y_pred / (1 - t1)
    temp2 = y_true * y_pred / t2
    numerator = exp_t(log_t(temp1, t1) + log_t(temp2, t2), 1)
    denominator = compute_normalization(y_pred, t2)
    loss_values = tf.reduce_sum(y_true * (-numerator / denominator), axis=-1)
    if label_smoothing > 0:
        smooth_loss = tf.reduce_sum(y_pred, axis=-1)
        loss_values = (1 - label_smoothing) * loss_values + label_smoothing * smooth_loss
    return tf.reduce_mean(loss_values)


# blob data for bi-tempered logistic loss from scikit-learn
from sklearn.datasets import make_blobs
import numpy as np
import random
X, y = make_blobs(n_samples=1000, centers=2, n_features=10, random_state=np.random.seed(50))
X = X.astype(np.float32)
y = y.astype(np.int32)

# create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
                loss=lambda y_true, y_pred: compute_bi_tempered_logistic_loss(y_pred, y_true, t1=0.8, t2=1.4, label_smoothing=0.1),
                metrics=['accuracy'])

# fit the model
# visually show the training process
import matplotlib.pyplot as plt
model.fit(X, y, epochs=100, verbose=1)

# evaluate the model and plot the results
loss, acc = model.evaluate(X, y, verbose=0)
print('Accuracy: %.3f' % acc)
plt.plot(model.history.history['loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

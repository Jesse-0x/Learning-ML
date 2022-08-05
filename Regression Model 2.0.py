import numpy as np
from random import randint
import matplotlib.pyplot as plt

x = [[11, -23, 24, -1, -19, 3, 0, 28, 8, -9, 28, -24, -24, -3, 9, 4, 18, -26, 3, 14, -18, -3, 20, -19, -30, -30, 9, 1, -29, 11],
     [13, 15, 15, -26, -1, 15, 13, -19, 1, 14, -3, 20, 13, -20, 30, -24, -11, -26, 9, -4, -21, 10, 8, 12, -7, -10, 7, 29, -17, 3],
     [-2, 22, -29, 2, -21, -6, -8, 27, -6, 7, 3, -20, -3, 3, 14, 4, 1, -12, -4, 17, -29, 21, -1, -6, -23, 13, -30, -10, -10, -28]]
y = [64578, -11072, 77992, -60749, -88718, 38460, 21600, 80302, 21800, 13349, 88206, -54409, -49846, -51913, 117719, -37421, 34638, -158875, 26858, 57861, -142256, 40649, 83193, -39723, -140830, -103441, 9488, 59886, -145004, 9044]

x = np.array(x)
y = np.array(y)


# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]


def zscore(x):
    return (x - np.mean(x)) / np.std(x)

# x = zscore(x)
# y = zscore(y)

py = []
def linear_regression(x, y, epochs=30000, learning_rate=0.003):
    m = len(x[0])
    d = len(x)
    w, b, tmp_w, su = np.array([0] * d), 0, [], 0
    for i in range(epochs):
        for j in range(d):
            tmp_w += [w[j] - learning_rate * (1 / m) * np.sum(((w.dot(x) + b) - y) * x[j])]
        tmp_b = b - learning_rate * (1 / m) * np.sum((w.dot(x) + b) - y)
        w = np.array(tmp_w)
        b = tmp_b
        tmp_w = []
        loss = (1 / 2 * m) * np.sum(((w.dot(x)) - y) ** 2)
        print(w, b, f'loss: {loss}')
        py.append(loss)
        if len(py) > 100 and py[-10] - loss == 0:
            break
    return w, b


w, b = linear_regression(x, y)
px = [i for i in range(len(py))]
plt.plot(px, py)
plt.show()
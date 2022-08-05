import numpy as np
import matplotlib.pyplot as plt

x1 = [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
x2 = [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [-166, -154, -142, -130, -118, -106, -94, -82, -70, -58, -46, -34, -22, -10, 2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122, 134, 146, 158, 170, 182]

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)


def linear_regression(x1, x2, y, epochs=5000, learning_rate=0.01):
    m = len(x1)
    x = np.array([x1, x2])
    w1, w2, b = 0, 0, 0
    for i in range(epochs):
        w = np.array([w1, w2])
        tmp_w1 = w1 - learning_rate * (1 / m) * np.sum(((w.dot(x) + b) - y) * x1)
        tmp_w2 = w2 - learning_rate * (1 / m) * np.sum(((w.dot(x) + b) - y) * x2)
        tmp_b = b - learning_rate * (1 / m) * np.sum((w.dot(x) + b) - y)
        w1 = tmp_w1
        w2 = tmp_w2
        b = tmp_b
        print(w, b)
    return w, b


def f(w1, w2, b, x1, x2):
    return w1 * x1 + w2 * x2 + b


w, b = linear_regression(x1, x2, y)
print(w, b)
print()

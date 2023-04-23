import numpy as np
import matplotlib.pyplot as plt
x = [1,2,3]
y = [2,4,6]

x = np.array(x)
y = np.array(y)

# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]


def linear_regression(x, y, epochs=50000, learning_rate=0.01):
    m = len(x)
    w, b = 0, 0
    for i in range(epochs):
        tmp_w = w - learning_rate * (1/m) * np.sum(((w * x + b) - y) * x)
        tmp_b = b - learning_rate * (1/m) * np.sum((w * x + b) - y)
        w = tmp_w
        b = tmp_b
        print(w, b)
    return w, b

def f(w, b):
    return w * x + b

w, b = linear_regression(x, y)
print(w, b)
print()
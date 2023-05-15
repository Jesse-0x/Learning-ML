import torch
import numpy as np
import matplotlib.pyplot as plt

# draw a graph of tensor
def draw_tensor_picture(tensor):
    # draw as a gray scale picture
    plt.imshow(tensor, cmap='gray')
    plt.show()

def flush_tensor_picture(tensor):
    # flush the picture
    plt.imshow(tensor, cmap='plasma')
    plt.draw()
    plt.pause(0.001)
    plt.clf()

while 1:
    my_tensor = torch.rand(100, 100)
    # draw_tensor_picture(my_tensor)
    # re process the tensor to draw a picutre of standard normal distribution

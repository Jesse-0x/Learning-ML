import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt


def line_to_fft(line):
    # no need for get the hz, just get the fft
    yf = scipy.fftpack.fft(line)
    # get the fft data
    yf = np.abs(yf[0:len(line)]) * 2 / (128 * len(line))
    return yf


if __name__ == '__main__':
    # get a data with a sine wave
    x = np.arange(0, 100, 2)
    y = np.sin(x)
    # convert the data to fft
    fft_y = line_to_fft(y)
    # plot the data
    plt.plot(x, fft_y)
    plt.show()

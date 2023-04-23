# give me a code that can read from microphone and output the sound wave in real time

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import scipy.fftpack

CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=0)

fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))
ax.set_ylim(0, 4096*8)
ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1200)
# ax.set_xlim(0, 1024)
plt.show(block=False)

while True:
    data = np.fromstring(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    # convert data to fft
    # yf = scipy.fftpack.fft(data)
    # data = np.abs(yf[0:CHUNK]) * 2 / (128 * CHUNK)
    # # fit data to plot
    # data = data - np.mean(data)
    # data = data / np.max(data) * 1000
    # # smooth out the data
    # data = np.convolve(data, np.ones((10,)) / 10, mode='same')
    # # data = np.flip(data, 0)

    # plot the data
    line.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()


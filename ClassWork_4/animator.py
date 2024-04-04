import os
from math import atan2
from random import shuffle

import cv2
import cmapy
import numpy as np
from numpy import pi, cos, sin
from scipy.fft import fft

def animate_fourier(fourier_coefficients, frames=100, interval=20):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')

    # Prepare the plot
    line, = ax.plot([], [], lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # Initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # Animation function: this is called sequentially
    def animate(i):
        N = len(fourier_coefficients)
        t = i / float(frames) * 2 * np.pi
        sums = np.sum([coeff * np.exp(1j * k * t) for k, coeff in enumerate(fourier_coefficients)], axis=0)

        x, y = sums.real, sums.imag
        line.set_data(x, y)
        time_text.set_text('Frame: %d' % i)
        return line, time_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)

    plt.show()


def animate(time_signal, top_perc=1.0, cmap='hsv'):

    freq_desc = fft(time_signal) / len(time_signal)  # complex array
    magnitudes = np.abs(freq_desc)
    phases = np.angle(freq_desc)
    time_steps = len(magnitudes)

    # dropping zero components for better color distribution
    tok_k_count = int(len(magnitudes) * top_perc)  # select the top k strongest components and neglecting the rest
    frequencies = np.sort(magnitudes.argsort()[-tok_k_count:][::-1])  # top k frequencies, frequency multiples actually
    magnitudes = magnitudes[frequencies]  # top k magnitudes
    phases = phases[frequencies]  # top k phases
    ############################################################
    # using the color map to color the components for a better visual experience
    colors = [cmapy.color(cmap, round(idx))
              for idx in np.linspace(0, 255, len(frequencies) * 2)
                ]  # linearly spaced colors
    shuffle(colors)
    ###########################################################
    # the animation simulation
    for time in range(time_steps + 100):  # extra 100 static frames in order to give some time for the viewer to see the full image in a video
        if time < time_steps:
            canvas = draw_components(magnitudes, phases, frequencies, time / time_steps, colors)
            cv2.imshow("Animation", canvas)
            if self.write_path:
                cv2.imwrite(os.path.join(self.write_path, f"{time:05d}.png"), canvas)
            cv2.waitKey(10)
        else:
            cv2.imwrite(os.path.join(self.write_path, f"{time:05d}.png"), canvas)

    print("PRESS ANY KEY TO EXIT")
    cv2.waitKey()

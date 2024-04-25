from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from signal_analysis.common import power_spectrum, set_custom_pyplot_styles

N = 2000
A = 5
f = 40
fs = 1000

dt = 1 / fs
times = dt * np.arange(N)
wave = A * np.sin(2 * np.pi * f * times) + A * 2 * np.random.randn(N)

freqs, spect = power_spectrum(signal=wave, fs=fs)

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [top, bot] = plt.subplots(2, 1)

    top.plot(times, wave)
    top.set_xlabel("Time [s]")
    top.set_ylabel("Amplitude")

    bot.plot(freqs, spect)
    bot.set_xlim(0, 150)
    bot.set_xlabel("Frequency [Hz]")
    bot.set_ylabel("Power")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    plt.show()

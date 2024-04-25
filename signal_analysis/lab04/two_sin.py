from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from signal_analysis.common import power_spectrum, set_custom_pyplot_styles
from signal_analysis.lab04.wave import fs, times

f1 = 11
f2 = 26
A1 = 2
A2 = 3
wave = A1 * np.sin(2 * np.pi * f1 * times) + A2 * np.sin(2 * np.pi * f2 * times)

freqs, spect = power_spectrum(signal=wave, fs=fs)


if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [top, bot] = plt.subplots(2, 1)

    top.plot(times, wave)
    top.set_title(f"${A1}\sin(2\pi\cdot{f1}t)+{A2}\sin(2\pi\cdot{f2}t)$")
    top.set_xlabel("Time [s]")
    top.set_ylabel("Amplitude")

    bot.plot(freqs, spect)
    bot.set_xlim(0, 60)
    bot.set_xlabel("Frequency [Hz]")
    bot.set_ylabel("Power")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    plt.show()

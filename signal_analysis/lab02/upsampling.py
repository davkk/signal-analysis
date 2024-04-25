from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from signal_analysis.common import power_spectrum, set_custom_pyplot_styles
from signal_analysis.lab02.spectra_comparison import A, f
from signal_analysis.lab02.spectra_comparison import fs as fs
from signal_analysis.lab02.spectra_comparison import infi as wave
from signal_analysis.lab02.spectra_comparison import time

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [[ltop, rtop], [lbot, rbot]] = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(f"Theoretical signal ${A}\\sin(2\\pi\\cdot{f}\\cdot t)$")

    ltop.set_xlabel("Time [s]")
    rtop.set_xlabel("Time [s]")
    lbot.set_xlabel("Frequency [Hz]")
    rbot.set_xlabel("Frequency [Hz]")

    ltop.set_ylabel("Amplitude [a.u.]")
    lbot.set_ylabel("Power [dB]")

    freq, power = power_spectrum(signal=wave, fs=fs)

    ltop.plot(time, wave)
    lbot.plot(freq, power)
    ltop.set_title("Signal without zero-padding")

    upsampled = np.zeros(wave.size * 2)
    upsampled[::2] = wave
    time = np.arange(wave.size * 2) / fs / 2

    freq, power = power_spectrum(signal=upsampled, fs=fs * 2)

    rtop.plot(time, upsampled)
    rbot.plot(freq, power)

    rtop.set_title("Signal with zero-padding")

    lbot.set_xscale("log")
    rbot.set_xscale("log")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

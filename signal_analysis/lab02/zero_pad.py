from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from signal_analysis.common import power_spectrum, set_custom_pyplot_styles
from signal_analysis.lab02.spectra_comparison import A, f
from signal_analysis.lab02.spectra_comparison import fs as fs
from signal_analysis.lab02.spectra_comparison import infi, time

# from signal_analysis.lab02.infi import fs, time, infi

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [[ltop, rtop], [lbot, rbot]] = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(f"Theoretical signal ${A}\\sin(2\\pi\\cdot{f}\\cdot t)$")

    lbot.set_xlabel("Frequency [Hz]")
    rbot.set_xlabel("Frequency [Hz]")
    ltop.set_xlabel("Time [s]")
    rtop.set_xlabel("Time [s]")
    ltop.set_ylabel("Amplitude [a.u.]")
    lbot.set_ylabel("Power [dB]")

    freq, power = power_spectrum(signal=infi, fs=fs)

    ltop.plot(time, infi)
    lbot.plot(freq[freq > 0], power[freq > 0])
    ltop.set_title("Signal without zero-padding")

    infi = np.pad(infi, (0, infi.size * 3))
    time = np.arange(infi.size) / fs

    freq, power = power_spectrum(signal=infi, fs=fs)

    rtop.plot(time, infi)
    rbot.plot(freq[freq > 0], power[freq > 0])

    rtop.set_title("Signal with zero-padding")

    lbot.set_xlim(0, 20)
    rbot.set_xlim(0, 20)

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

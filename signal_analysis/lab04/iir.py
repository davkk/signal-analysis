from pathlib import Path

import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

from signal_analysis.common import (
    power_spectrum,
    set_custom_pyplot_styles,
    signaltonoise,
)
from signal_analysis.lab04.wave import freqs as freqs0
from signal_analysis.lab04.wave import fs
from signal_analysis.lab04.wave import spect as spect0
from signal_analysis.lab04.wave import times as times0
from signal_analysis.lab04.wave import wave as wave0

nyq = fs / 2
order = 4

filters = {
    "highpass": signal.butter(order, 42 / nyq, btype="highpass"),
    "lowpass": signal.butter(order, 42 / nyq, btype="lowpass"),
    "bandpass": signal.butter(
        order, [38 / nyq, 42 / nyq], btype="bandpass"
    ),
}

if __name__ == "__main__":
    set_custom_pyplot_styles()

    for filter_name, (b, a) in filters.items():
        fig, [top, mid, bot] = plt.subplots(3, 1)

        result = signal.lfilter(b, a, wave0)

        w, h = signal.freqz(b, a, worN=2000)
        top.plot(w * nyq / np.pi, np.abs(h))
        top.set_title("Filter frequency spectrum")
        top.set_xlabel("Frequency [Hz]")
        top.set_ylabel("Power")

        mid.plot(times0, wave0, c="#cacecd", linewidth=1.0, label="before")
        mid.plot(times0, result, label="after")
        mid.set_title("Result")
        mid.legend(loc="lower right")
        mid.set_xlabel("Time [s]")
        mid.set_ylabel("Amplitude")

        freqs, spect = power_spectrum(signal=result, fs=fs)

        bot.plot(freqs0, spect0, c="#cacecd", linewidth=1.0, label="before")
        bot.plot(freqs, spect, label="after")
        bot.set_xlim(0, 150)
        bot.set_xticks(np.arange(0, 150, 20))
        bot.set_xlabel("Frequency [Hz]")
        bot.set_ylabel("Power")
        bot.set_yscale("log")
        bot.set_title("Spectrum of the result")
        bot.legend(loc="lower right")

        SNR = 10 * np.log10(signaltonoise(np.abs(result)))
        fig.suptitle(f"{SNR=:.3f} [dB]")
        fig.tight_layout()
        plt.savefig(Path(__file__).with_suffix(f".{filter_name}.pdf"))
        # plt.show()

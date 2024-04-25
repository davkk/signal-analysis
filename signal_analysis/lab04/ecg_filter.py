from pathlib import Path

import numpy as np
import scipy
import scipy.signal as signal
from matplotlib import pyplot as plt

from signal_analysis.common import power_spectrum, set_custom_pyplot_styles

fs = 500
nyq = fs / 2
wave = scipy.io.loadmat("data/ecg.mat")["ecg"].flatten()
times0 = np.arange(wave.size) / fs

freqs0, spect0 = power_spectrum(signal=wave, fs=fs)

order = 4
b, a = signal.butter(order, [0.9 / nyq, 15 / nyq], btype="bandpass")

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [top, mid, bot] = plt.subplots(3, 1)

    result = signal.lfilter(b, a, wave)

    w, h = signal.freqz(b, a, worN=2000)
    top.plot(w * nyq / np.pi, np.abs(h))
    top.set_title("Spectrum of the filter")
    top.set_xlabel("Frequency [Hz]")
    top.set_ylabel("Power")
    top.set_xlim(0, 40)

    mid.plot(times0, wave, c="#cacecd", linewidth=1.0, label="before")
    mid.plot(times0, result, label="after")
    mid.set_title("Result")
    mid.legend(loc="lower right")
    mid.set_xlabel("Time [s]")
    mid.set_ylabel("Amplitude")

    freqs, spect = power_spectrum(signal=result, fs=fs)

    bot.plot(freqs0, spect0, c="#cacecd", linewidth=1.0, label="before")
    bot.plot(freqs, spect, label="after")
    bot.set_xlim(0, 40)
    bot.set_xlabel("Frequency [Hz]")
    bot.set_ylabel("Power")
    bot.set_title("Spectrum of the result")
    bot.legend(loc="lower right")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    plt.show()

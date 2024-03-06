from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from sat.common import set_custom_pyplot_styles
from sat.lab01.fourier import freqs, spectrum_db

peaks, _ = signal.find_peaks(spectrum_db, distance=100)
peaks = peaks[np.argsort(spectrum_db[peaks])[-11:]]

# range 8 * 12 = 39 + 57
min_note = -57
max_note = 39

base_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

tone_freqs = 440 * np.power(2, np.arange(min_note, max_note + 1) / 12)
tone_names = [
    *[
        f"{base_names[halftone]}{octave}"
        for octave in range(8)
        for halftone in range(12)
    ],
    "C8",
]

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 8))

    ax.scatter(freqs[peaks], spectrum_db[peaks], c="orange")
    ax.semilogy(freqs, spectrum_db)

    for freq_peak, peak_value in zip(freqs[peaks], spectrum_db[peaks]):
        idx = np.abs(tone_freqs - freq_peak).argmin()
        ax.annotate(
            tone_names[idx],
            xy=(freq_peak, peak_value),
            xytext=(0, 10),
            textcoords="offset pixels",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.axvline(x=freq_peak, c="gray", alpha=0.5, linestyle="--")

    ax.set_xticks(freqs[peaks])

    ax.set_xlim(-50 + sorted(freqs[peaks])[0], 50 + sorted(freqs[peaks])[-1])
    ax.set_ylim(40, 90)

    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel(r"Power [dB]")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

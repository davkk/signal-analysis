from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sat.common import set_custom_pyplot_styles
from sat.lab01.wave import sample_rate, wave

spectrum = np.fft.fft(wave)
freqs = np.fft.fftfreq(wave.size, 1 / sample_rate)

f_range = (freqs >= 16) & (freqs <= 4e3)
freqs = freqs[f_range]
spectrum = spectrum[f_range]

spectrum_db = 10 * np.log10(np.abs(spectrum) + 1e-15)

if __name__ == "__main__":
    set_custom_pyplot_styles()

    plt.loglog(freqs, spectrum_db)

    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"Power [dB]")

    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

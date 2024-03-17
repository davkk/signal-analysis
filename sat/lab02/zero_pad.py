from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sat.common import power_spec, set_custom_pyplot_styles
from sat.lab02.spectra_comparison import A, f, infi
from sat.lab02.spectra_comparison import fs as fs
from sat.lab02.spectra_comparison import time

# from sat.lab02.infi import fs, time, infi

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [[ltop, rtop], [lbot, rbot]] = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(f"Theoretical signal ${A}\\sin(2\\pi\\cdot{f}\\cdot t)$")

    rbot.set_xlabel("Frequency [Hz]")
    ltop.set_ylabel("Amplitude [a.u.]")
    lbot.set_xlabel("Time [s]")
    lbot.set_ylabel("Power [dB]")

    freq, power = power_spec(signal=infi, sr=fs)

    ltop.plot(time, infi)
    lbot.plot(freq[freq > 0], power[freq > 0])
    ltop.set_title("Signal without zero-padding")

    infi = np.pad(infi, (0, infi.size * 2))
    time = np.arange(infi.size) / fs

    freq, power = power_spec(signal=infi, sr=fs)

    rtop.plot(time, infi)
    rbot.plot(freq[freq > 0], power[freq > 0])

    rtop.set_title("Signal with zero-padding")

    # fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from sat.common import set_custom_pyplot_styles
from sat.lab03.chirp import fs, wave
from sat.lab03.cohen_utils import cohen

sxx, freq, time = cohen(signal.hilbert(wave), fs)

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, [left, right] = plt.subplots(ncols=2)

    left.pcolormesh(time, freq, np.abs(sxx), shading="gouraud")
    left.set_title("Wigner-Ville transform (abs+linear)")
    left.set_xlabel("Time [s]")
    left.set_ylabel("Frequency [Hz]")

    right.pcolormesh(time, freq, np.log10(np.abs(sxx)), shading="gouraud")
    right.set_title("Wigner-Ville transform (abs+log)")
    right.set_xlabel("Time [s]")
    right.set_ylabel("Frequency [Hz]")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"))
    # plt.show()

from pathlib import Path

import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

from sat.common import set_custom_pyplot_styles

fs = 500


def get_chirp(*, N: int, f0=20, f1=100):
    time = np.arange(1, N + 1, step=1) / fs
    wave = signal.chirp(time, f0=f0, t1=N / fs, f1=f1)
    return time, wave


time, wave = get_chirp(N=512)

if __name__ == "__main__":
    set_custom_pyplot_styles()

    plt.plot(time, wave)

    plt.tight_layout()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [a.u.]")

    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

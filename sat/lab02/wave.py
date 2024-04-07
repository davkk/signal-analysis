from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from sat.common import set_custom_pyplot_styles

fs, wave = scipy.io.wavfile.read("./data/windows.wav")
time = np.arange(wave.size) / fs

trim = (time > 0.2) & (time < 3)
time = time[trim]
wave = wave[trim]

if __name__ == "__main__":
    set_custom_pyplot_styles()

    trim = (time > 0.2) & (time < 3)

    plt.plot(time[trim], wave[trim])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [a.u.]")

    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

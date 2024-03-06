from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from sat.common import set_custom_pyplot_styles

sample_rate, wave = scipy.io.wavfile.read("./data/chord.wav")
print(f"{sample_rate=}")

wave = wave[int(8.6e4) :]
time = np.arange(wave.size) / sample_rate

if __name__ == "__main__":
    set_custom_pyplot_styles()

    plt.plot(time, wave)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".pdf"))
    # plt.show()

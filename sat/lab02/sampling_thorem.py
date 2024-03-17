from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from sat.common import power_spec, set_custom_pyplot_styles
from sat.lab02.wave import fs, time, wave

fss = fs // np.array([1, 1.5, 2, 5, 10])

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig = plt.figure(figsize=(10, 14), dpi=100, constrained_layout=True)
    subfigs = fig.subfigures(nrows=5, ncols=1)

    for idx, curr_fs in enumerate(fss):
        subfig = subfigs[idx]
        [left, right] = subfig.subplots(nrows=1, ncols=2)

        if idx == 0:
            subfig.suptitle(f"Original sample rate $f_s={curr_fs}$ [Hz]")
        elif idx == 2:
            subfig.suptitle(f"$f_s={curr_fs}$ [Hz] (Nyquist)")
        else:
            subfig.suptitle(f"$f_s={curr_fs}$ [Hz]")

        samples = round(wave.size * curr_fs / fs)
        rwave, rtime = signal.resample(wave, samples, t=time)

        if idx > 0:
            left.plot(time, wave, c="lightgray")

        left.plot(rtime, rwave, ".")
        left.set_xlabel("Time [s]")
        left.set_ylabel("Amplitude [a.u.]")
        left.set_xlim(1, 1.1)

        freq, power = power_spec(
            signal=rwave,
            sr=curr_fs,
        )
        positive = freq > 0

        right.plot(freq[positive], power[positive])
        right.set_xlabel("Frequency [Hz]")
        right.set_ylabel("Power [dB]")
        right.set_yticks(np.arange(20, 81, 20))

    plt.savefig(Path(__file__).with_suffix(".pdf"), dpi=300)
    # plt.show()

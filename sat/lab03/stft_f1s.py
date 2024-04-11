from pathlib import Path

import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

from sat.lab03.chirp import fs, get_chirp
from sat.lab03.cohen_utils import cohen

if __name__ == "__main__":
    f1s = [200, 300, 400, 500]
    nfft = 128

    fig, axs = plt.subplots(ncols=2, nrows=len(f1s), figsize=(9, 13))

    for idx, f1 in enumerate(f1s):
        [wv, sp] = axs[idx]
        time, wave = get_chirp(N=512, f1=f1)

        sxx, f, t = cohen(signal.hilbert(wave), fs)
        wv.pcolormesh(t, f, np.log10(np.abs(sxx)), shading="gouraud")
        wv.set_title(f"W-V transform, $f_1={f1}$ [Hz]")
        wv.set_xlabel("Time [s]")
        wv.set_ylabel("Frequency [Hz]")

        f, t, sxx = signal.spectrogram(
            wave,
            fs=fs,
            window=("hamming"),
            nperseg=nfft // 2,
            nfft=nfft,
        )
        sp.pcolormesh(t, f, np.log10(np.abs(sxx)), shading="gouraud")
        sp.set_title(f"STFT, $f_1={f1}$ [Hz]")
        sp.set_xlabel("Time [s]")
        sp.set_ylabel("Frequency [Hz]")

    fig.suptitle(f"$\\text{{{nfft=}}}$")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"))
    # plt.show()

from pathlib import Path

import scipy.signal as signal
from matplotlib import pyplot as plt

from signal_analysis.common import set_custom_pyplot_styles
from signal_analysis.lab03.chirp import fs, get_chirp

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    axs = axs.flatten()

    nfft = 512

    for idx, N in enumerate([1024, 2048]):
        ax = axs[idx]

        _, wave = get_chirp(N=N)
        freqs, times, spect = signal.spectrogram(
            wave,
            fs=fs,
            window=("hamming"),
            nperseg=nfft // 2,
            nfft=nfft,
        )

        ax.pcolormesh(times, freqs, spect, shading="gouraud")

        ax.set_title(f"$N = {N // nfft}\\cdot{nfft} = {N}$")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"))
    # plt.show()

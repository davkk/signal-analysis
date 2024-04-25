from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

from signal_analysis.common import set_custom_pyplot_styles
from signal_analysis.lab03.chirp import fs, get_chirp
from signal_analysis.lab03.cohen_utils import cohen


def layout():
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    fig = plt.figure()

    top = plt.subplot(gs[0, :])
    left = plt.subplot(gs[1, :1])
    right = plt.subplot(gs[1, 1:])

    return fig, top, left, right


def wv_stft(*, wave, fs, wv, stft, nfft):
    sxx, f, t = cohen(signal.hilbert(wave), fs)
    wv.pcolormesh(t, f, np.log10(np.abs(sxx)), shading="gouraud")
    wv.set_title("W-V transform")
    wv.set_xlabel("Time [s]")
    wv.set_ylabel("Frequency [Hz]")

    f, t, sxx = signal.spectrogram(
        wave,
        fs=fs,
        window=("hamming"),
        nperseg=nfft // 2,
        nfft=nfft,
    )
    stft.pcolormesh(t, f, np.log10(np.abs(sxx)), shading="gouraud")
    stft.set_title("STFT")
    stft.set_xlabel("Time [s]")
    stft.set_ylabel("Frequency [Hz]")


if __name__ == "__main__":
    set_custom_pyplot_styles()

    N = 512
    nfft = 128
    f0 = 20
    f1 = 200

    time, chirp1 = get_chirp(N=N, f0=f0, f1=f1)
    _, chirp2 = get_chirp(N=N, f0=2 * f0, f1=f0 + f1)
    wave = chirp1 + chirp2

    fig, top, wv, stft = layout()

    top.plot(time, wave)
    top.set_title("Two chirps changing in parallel")
    top.set_xlabel("Time [s]")
    top.set_ylabel("Amplitude [a. u.]")

    wv_stft(
        wave=chirp1 + chirp2,
        fs=fs,
        wv=wv,
        stft=stft,
        nfft=nfft,
    )

    fig.suptitle(f"$\\text{{{nfft=}}}$, ${N=}$")

    fig.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"))
    # plt.show()

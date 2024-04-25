from pathlib import Path

from matplotlib import pyplot as plt
from signal_analysis.common import set_custom_pyplot_styles

from signal_analysis.lab03.chirp import fs, get_chirp
from signal_analysis.lab03.stft_parallel import layout, wv_stft

if __name__ == "__main__":
    set_custom_pyplot_styles()

    N = 512
    nfft = 128
    f0 = 20
    f1 = 200

    time, chirp1 = get_chirp(N=N, f0=f0, f1=f1)
    _, chirp2 = get_chirp(N=N, f0=2 * f0, f1=2 * f1)
    wave = chirp1 + chirp2

    fig, top, wv, stft = layout()

    top.plot(time, wave)
    top.set_title("Two chirps changing not in parallel")
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

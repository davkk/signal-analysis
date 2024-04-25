from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from signal_analysis.common import power_spectrum, set_custom_pyplot_styles

# jaki sygnal
# wykres syg widmo mocy
# jaki window wykres i widmo
# przed i po decymacji (wycinanie co ktores) jak zmienia sie widmo mocy

N = 1000
A = 5
f = 10
fs = 1000

dt = 1 / fs
time = dt * np.arange(N)

infi = A * np.sin(2 * np.pi * f * time)
window = time[time < 5 / f]
rect = np.pad(np.full(window.size, 1), ((time.size - window.size) // 2))
nonrect = scipy.signal.windows.gaussian(N, std=1 / f * N)

funcs = {
    "Infinite signal": infi,
    "Rectangular window": rect,
    "Finite signal": infi * rect,
    "Non-rectangular window": nonrect,
    "Smoothed finite signal": infi * nonrect,
}

if __name__ == "__main__":
    set_custom_pyplot_styles()

    fig = plt.figure(figsize=(8, 12), dpi=300, constrained_layout=True)
    subfigs = fig.subfigures(nrows=5, ncols=1)

    for idx, (title, signal) in enumerate(funcs.items()):
        subfig = subfigs[idx]
        [left, right] = subfig.subplots(nrows=1, ncols=2)

        subfig.suptitle(title)

        left.plot(time, signal)
        left.set_xlabel("Time [s]")
        left.set_ylabel("Amplitude [a.u.]")

        freqs, spectrum = power_spectrum(signal=signal, fs=fs)
        f_range = (freqs >= -30) & (freqs <= 30)

        right.plot(freqs[f_range], spectrum[f_range])
        right.set_xlabel("Frequency [Hz]")
        right.set_ylabel("Power [dB]")
        right.set_xticks(np.arange(-30, 31, 10))

    plt.savefig(Path(__file__).with_suffix(".pdf"), dpi=300)
    # plt.show()

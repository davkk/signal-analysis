from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def set_custom_pyplot_styles():
    # custom pyplot styles
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["axes.formatter.limits"] = -3, 3
    plt.rcParams["axes.grid"] = False
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["axes.formatter.use_mathtext"] = True

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["*", "1", "+", "2", ".", "3"]
    return colors, markers


def power_spectrum(
    *, signal: npt.NDArray, fs: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    spectrum = np.fft.fft(signal)

    spectrum = np.abs(spectrum) ** 2
    spectrum = np.roll(spectrum, spectrum.size // 2)

    freqs = np.fft.fftfreq(signal.size, 1 / fs)
    freqs = np.roll(freqs, freqs.size // 2)

    return freqs, spectrum

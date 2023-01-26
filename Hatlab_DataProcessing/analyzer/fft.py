import warnings
import numpy as np
import scipy as sp
import h5py
from scipy.integrate import odeint
from scipy.fft import fft as scifft, fftfreq

import matplotlib.pyplot as plt
from Hatlab_DataProcessing.helpers.unit_converter import t2f


def fft(tList, data, tUnit=None, plot=True, plot_ax=None):
    N = len(tList)
    T = tList[1] -tList[0]
    F_data = scifft(data)
    F_data = 2.0 / N * np.abs(F_data[0:N // 2])
    F_freq = fftfreq(N, T)[:N // 2]
    if plot:
        if plot_ax is None:
            fig, plot_ax = plt.subplots()
        plot_ax.plot(F_freq, F_data)
        plot_ax.set_yscale("log")
        if tUnit is not None:
            plot_ax.set_xlabel(f"Freq {t2f(tUnit)}")
        plot_ax.grid(True)

    return F_freq, F_data
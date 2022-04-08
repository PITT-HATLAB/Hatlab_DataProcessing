import warnings
import numpy as np
import scipy as sp
import h5py
from scipy.integrate import odeint
from scipy.fft import fft as scifft, fftfreq

import matplotlib.pyplot as plt
from Hatlab_DataProcessing.helpers.unit_converter import t2f


def fft(tList, data, tUnit=None, plot=True):
    N = len(tList)
    T = tList[1] -tList[0]
    F_data = scifft(data)
    F_data = 2.0 / N * np.abs(F_data[0:N // 2])
    F_freq = fftfreq(N, T)[:N // 2]
    if plot:
        plt.figure()
        plt.plot(F_freq, F_data)
        plt.yscale("log")
        plt.grid()
        if tUnit is not None:
            plt.xlabel(f"Freq {t2f(tUnit)}")

    return F_freq, F_data
from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
import h5py
from Hatlab_DataProcessing.fitter.fitter_base import Fit, FitResult
from Hatlab_DataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
TWOPI = 2 * np.pi
PI = np.pi




def getVNAData(filename, freq_unit='Hz', plot=1, trim=0):
    trim_end = None if trim==0 else -trim
    f = h5py.File(filename, 'r')
    freq = f['Freq'][()][trim: trim_end] * freqUnit(freq_unit)
    phase = f['S21'][()][1] [trim: trim_end] / 180 * np.pi
    mag = f['S21'][()][0][trim: trim_end]
    f.close()

    lin = 10 ** (mag / 20.0)
    real = lin * np.cos(phase)
    imag = lin * np.sin(phase)

    if plot:
        plt.figure('mag')
        plt.plot(freq / 2 / np.pi, mag)
        plt.figure('phase')
        plt.plot(freq / 2 / np.pi, phase)

    return (freq, real, imag, mag, phase)


class CavReflectionResult():
    def __init__(self, lmfit_result:lmfit.model.ModelResult):
        self.lmfit_result=lmfit_result
        self.params = lmfit_result.params
        self.f0 = self.params["f0"].value
        self.Qext = self.params["Qext"].value
        self.Qint = self.params["Qint"].value
        self.Qtot = self.Qext * self.Qint / (self.Qext + self.Qint)
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, **figArgs):
        real_fit = self.lmfit_result.best_fit.real
        imag_fit = self.lmfit_result.best_fit.imag
        mag_fit, phase_fit = realImag2magPhase(real_fit, imag_fit)
        mag_data, phase_data = realImag2magPhase(self.lmfit_result.data.real, self.lmfit_result.data.imag)

        fig_args_ = dict(figsize=(12, 5))
        fig_args_.update(figArgs)
        plt.figure(**fig_args_)
        plt.subplot(1, 2, 1)
        plt.title('mag (dB pwr)')
        plt.plot(self.freqData, mag_data, '.')
        plt.plot(self.freqData, mag_fit)
        plt.subplot(1, 2, 2)
        plt.title('phase')
        plt.plot(self.freqData, phase_data, '.')
        plt.plot(self.freqData, phase_fit)
        plt.show()

    def print(self):
        print(f'f (Hz): {rounder(self.f0, 9)}+-{rounder(self.params["f0"].stderr, 9)}')
        print(f'Qext: {rounder(self.Qext, 5)}+-{rounder(self.params["Qext"].stderr, 5)}')
        print(f'Qint: {rounder(self.Qint, 5)}+-{rounder(self.params["Qint"].stderr, 5)}')
        print('Q_tot: ', rounder(self.Qtot, 5))
        print('T1 (s):', rounder(self.Qtot / self.f0 / 2 / np.pi, 5), '\nMaxT1 (s):',
              rounder(self.Qint / self.f0 / 2 / np.pi, 5))
        print('kappa/2Pi: ', rounder(self.f0 / self.Qtot / 1e6), 'MHz')


class CavReflection(Fit):
    @staticmethod
    def model(coordinates, Qext, Qint, f0, magBack, phaseOff) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        omega0 = f0 * TWOPI
        delta = coordinates * TWOPI - omega0
        S_11_nume = 1 - Qint / Qext + 1j * 2 * Qint * delta / omega0
        S_11_denom = 1 + Qint / Qext + 1j * 2 * Qint * delta / omega0
        S11 = magBack * (S_11_nume / S_11_denom) * np.exp(1j * (phaseOff))

        realPart = np.real(S11)
        imagPart = np.imag(S11)
        return realPart + 1j * imagPart

    @staticmethod
    def guess(coordinates, data):
        freq = coordinates
        phase = np.unwrap(np.angle(data))
        mag = np.abs(data)
        real = np.real(data)
        imag = np.imag(data)

        f0Guess = freq[np.argmin(mag)]  # smart guess of "it's probably the lowest point"
        magBackGuess = np.average(mag[:int(len(freq) / 5)])
        QextGuess = 1e5
        QintGuess = 1e5
        phaseOffGuess = phase[np.argmin(mag)]

        Qext = lmfit.Parameter("Qext", value=QextGuess, min=QextGuess/100, max=QextGuess*100)
        Qint = lmfit.Parameter("Qint", value=QintGuess, min=QintGuess/100, max=QintGuess*100)
        f0 = lmfit.Parameter("f0", value=f0Guess, min=freq[0], max=freq[-1])
        magBack = lmfit.Parameter("magBack", value=magBackGuess, min=magBackGuess/1.1, max=magBackGuess*1.1)
        phaseOff = lmfit.Parameter("phaseOff", value=phaseOffGuess, min=-PI, max=PI)

        return dict(Qext=Qext, Qint=Qint, f0=f0, magBack=magBack, phaseOff=phaseOff)

    def run(self, *args: Any, **kwargs: Any) -> CavReflectionResult:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return CavReflectionResult(lmfit_result)

if __name__ == '__main__':
    # filepath = easygui.fileopenbox()
    filepath = r'L:\Data\WISPE3D\Modes\20210809\CavModes\Cav'
    (freq, real, imag, mag, phase) = getVNAData(filepath, plot=1)

    cavRef = CavReflection(freq, real+1j*imag)
    results = cavRef.run()



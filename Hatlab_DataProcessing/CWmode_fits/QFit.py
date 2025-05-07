import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py
import inspect
from scipy.optimize import curve_fit
# import easygui
from plottr.data import datadict_storage as dds, datadict as dd
from plottr.data.datadict_storage import all_datadicts_from_hdf5

'''
The point of this file is not to give mediocre fits, but to generate a good initial guess for a better fitter. 
Fitting functions often don't converge without good initial guesses, but tuning the initial guesses can be tedious.
Fortunately, they can be somewhat automated.
'''

FREQ_UNIT = {'GHz': 1e9,
             'MHz': 1e6,
             'KHz': 1e3,
             'Hz': 1.0
             }

def rounder(value):
    return "{:.4e}".format(value)


def reflectionFunc(freq, Qext, Qint, f0, A, B, C, D):
    '''
    A is the phase offset. B is the rate of linear change of phase offset (proportional to electrical delay)
    C is the magnitude. D is the rate of linear change in magnitude, to account for small changes.
    '''
    omega0 = f0
    delta = freq - omega0
    S_11_up = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) - Qext / omega0
    S_11_down = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) + Qext / omega0
    S11 = (S_11_up / S_11_down) * np.exp(1j * (A + B*(freq-freq[0]) ) ) * (C + D*(freq-freq[0]))
    realPart = np.real(S11)
    imagPart = np.imag(S11)

    return (realPart + 1j * imagPart).view(np.float64)
    # return realPart
    # return imagPart


def reflectionFunc_re(freq, Qext, Qint, f0, magBack, A, B):
    return reflectionFunc(freq, Qext, Qint, f0, magBack,  A, B)[::2]


def getData_from_datadict(filepath, plot_data=None):
    datadict = all_datadicts_from_hdf5(filepath)['data']
    powers_dB = datadict.extract('power')['power']['values']
    freqs = datadict.extract('power')['frequency']['values'] * 2 * np.pi
    phase_rad = datadict.extract('phase')['phase']['values'] * np.pi / 180

    lin = np.power(10, powers_dB / 20)
    real = lin * np.cos(phase_rad)
    imag = lin * np.sin(phase_rad)

    print(np.size(phase_rad))
    print(np.size(phase_rad))

    if plot_data:
        plt.figure('mag')
        plt.plot(freqs, powers_dB)
        plt.figure('phase')
        plt.plot(freqs, phase_rad)

    return (freqs, real, imag, powers_dB, phase_rad)


def getData(filename, method='hfss', freq_unit='GHz', plot_data=1):
    if method == 'hfss':
        """The csv file must be inthe format of:
            freq  mag(dB)  phase(cang_deg)  
        """
        with open(filename) as csvfile:
            csvData = list(csv.reader(csvfile))
            csvData.pop(0)  # Remove the header
            data = np.zeros((len(csvData[0]), len(csvData)))
            for x in range(len(csvData)):
                for y in range(len(csvData[0])):
                    data[y][x] = csvData[x][y]

        freq = data[0] * 2 * np.pi * FREQ_UNIT[freq_unit]  # omega
        phase = np.array(data[2]) / 180. * np.pi
        mag = data[1]
        lin = 10 ** (mag / 20.0)

    elif method == 'vna':
        f = h5py.File(filename, 'r')
        freq = f['VNA Frequency (Hz)'][()] * 2 * np.pi
        phase = f['Phase (deg)'][()]
        mag = f['Power (dB)'][()]
        lin = 10 ** (mag / 20.0)
        f.close()

    elif method == 'vna_old':
        f = h5py.File(filename, 'r')
        freq = f['Freq'][()] * 2 * np.pi
        phase = f['S21'][()][1] / 180. * np.pi
        mag = f['S21'][()][0]
        lin = 10 ** (mag / 20.0)
        f.close()

    else:
        raise NotImplementedError('method not supported')

    imag = lin * np.sin(phase)
    real = lin * np.cos(phase)

    if plot_data:
        plt.figure('mag')
        plt.plot(freq / 2 / np.pi, mag)
        plt.figure('phase')
        plt.plot(freq / 2 / np.pi, phase)

    return (freq, real, imag, mag, phase)

    # if method == 'vna':
    #     f = h5py.File(filename,'r')
    #     freq = f['VNA Frequency (Hz)'][()]
    #     phase = f['Phase (deg)'][()] / 180. * np.pi
    #     lin = 10**(f['Power (dB)'][()] / 20.0)
    # if method == 'vna_old':
    #     f = h5py.File(filename,'r')
    #     freq = f['Freq'][()]
    #     phase = f['S21'][()][0] / 180. * np.pi
    #     lin = 10**(f['S21'][()][1] / 20.0)


def fit(freq, real, imag, mag, phase, real_only=0, bounds=None, QextGuess=None, QintGuess=None,
            AGuess=None, BGuess=None, CGuess=None, DGuess=None, plot=False, printout=False, n=3):

    if QextGuess == None:
        QextGuess = np.mean(freq)/(10*(freq[1]-freq[0]))
    if QintGuess == None:
        QintGuess = np.mean(freq)/(10*(freq[1]-freq[0]))

    if AGuess == None:
        AGuess = np.angle(real[0] + 1j * imag[0]) + np.pi
    if BGuess == None:
        BGuess = 0
    if CGuess == None:
        CGuess = np.mean(np.abs(real + 1j * imag))
    if DGuess == None:
        DGuess = 0

    S21 = real + 1j * imag
    f0Guess = rough_guess(freq, S21, n=n)

    if bounds == None:
        bounds = ([QextGuess / 10.0, QintGuess / 10.0, f0Guess / 1.1, -2 * np.pi, -1e-6, CGuess / 2.0, -CGuess * 1e-6],
                  [QextGuess * 10.0, QintGuess * 10.0, f0Guess * 1.1, 2 * np.pi, 1e-6, CGuess * 2.0, CGuess * 1e-6])

    target_func = reflectionFunc
    data_to_fit = (real + 1j * imag).view(np.float64)
    if real_only:
        target_func = reflectionFunc_re
        data_to_fit = real
    popt, pcov = curve_fit(target_func, freq, data_to_fit,
                           p0=(QextGuess, QintGuess, f0Guess, AGuess, BGuess, CGuess, DGuess),
                           bounds=bounds,
                           maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)

    if printout:
        print(f'f (Hz): {rounder(popt[2] / 2 / np.pi)}', )
        fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
        for i in range(2):
            print(f'{fitting_params[i]}: {rounder(popt[i])} +- {rounder(np.sqrt(pcov[i, i]))}')
        Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
        print('Q_tot: ', rounder(Qtot), '\nT1 (s):', rounder(Qtot / popt[2]),
              f"Kappa: {rounder(popt[2] / 2 / np.pi / Qtot)}", )

        print(f"Kappa ext: {rounder(popt[2] / 2 / np.pi / popt[0])}",
              f"\nKappa int: {rounder(popt[2] / 2 / np.pi / popt[1])}", )

    if plot:
        popt_guess = [QextGuess, QintGuess, f0Guess, AGuess, BGuess, CGuess, DGuess]

        plotRes(freq, real, imag, mag, phase, popt_guess)
        plotRes(freq, real, imag, mag, phase, popt)

        plt.figure()
        S21d = decimate_by_two(S21, n=n)
        metric = dx2(np.imag(S21d)) ** 2 + dx2(np.real(S21d)) ** 2
        plt.plot(metric)

    return popt, pcov

def fit_mode_from_ddh5(filepath, plot=False, printout=False):

    (freq, real, imag, mag, phase) = getData_from_datadict(filepath, plot_data=0)
    popt, pcov = fit(freq, real, imag, mag, phase, plot=plot, printout=printout)

    return popt, pcov


def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq / (2 * np.pi)
    realRes = reflectionFunc(freq, *popt)[::2]
    imagRes = reflectionFunc(freq, *popt)[1::2]
    # realRes = reflectionFunc(freq, *popt)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('real')
    plt.plot(xdata, real, '.')
    plt.plot(xdata, realRes)
    plt.subplot(1, 2, 2)
    plt.title('imag')
    plt.plot(xdata, imag, '.')
    plt.plot(xdata, imagRes)
    plt.show()


def decimate_by_two(x, n=1):
    xc = x.copy()
    xd = xc.copy()

    for i in range(0, n):

        if len(xc) // 2 != len(xc) / 2:
            xc = np.concatenate([xc, np.array([xc[-1]])])

        xc = np.reshape(xc, (len(xc) // 2, 2))
        xd = np.mean(xc, axis=1)

        xc = xd.copy()

    return xd


def dx2(x):
    xc = np.zeros(len(x) + 2)
    xc[1:-1] = x
    xc[0] = xc[1]
    xc[-1] = xc[-2]

    dx2 = x * 0
    dx2 -= xc[0:-2]
    dx2 += 2 * xc[1:-1]
    dx2 -= xc[2:]
    return dx2

def rough_guess(freq, S21, n=2):
    S21d = decimate_by_two(S21, n=n)
    freq = decimate_by_two(freq, n=n)

    metric = dx2(np.imag(S21d)) ** 2 + dx2(np.real(S21d)) ** 2

    f0_guess = freq[np.argmax(metric)]
    return f0_guess


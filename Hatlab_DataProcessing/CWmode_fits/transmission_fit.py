import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py
import inspect
from scipy.optimize import curve_fit
# import easygui
from plottr.data import datadict_storage as dds, datadict as dd
from plottr.data.datadict_storage import all_datadicts_from_hdf5


FREQ_UNIT = {'GHz': 1e9,
             'MHz': 1e6,
             'KHz': 1e3,
             'Hz': 1.0
             }

def rounder(value):
    return "{:.4e}".format(value)


def transmission_func(freq, Qext, Qint, f0):
    '''
    This is for the S21 parameter in transmission. The Qext is assumed to be the external Q for BOTH the left and 
    right port (you can't determine a difference in left and right from just S21).
    If they are differeny, Qext is just the geometric average.
    
    A is the phase offset. B is the rate of linear change of phase offset (proportional to electrical delay)
    C is the magnitude. D is the rate of linear change in magnitude, to account for small changes.
    '''
    omega0 = f0*2*np.pi
    delta = (freq - f0)*2*np.pi
    S_21_up = - 2*Qint/Qext
    S_21_down = 1 - 2j*Qint * (delta/omega0)
    S21 = (S_21_up / S_21_down)

    return np.abs(S21)

def getData_from_datadict(filepath, plot_data=None, freqs_name='frequency', phase_name='phase',
                                 mag_name='power', use_radians=True):
    datadict = all_datadicts_from_hdf5(filepath)['data']
    powers_dB = datadict.extract('power')['power']['values']
    freqs = datadict.extract('power')[freqs_name]['values']
    phase = datadict.extract('phase')[phase_name]['values']

    if use_radians == False:
        phase = phase * np.pi / 180

    lin = np.power(10, powers_dB / 20)
    real = lin * np.cos(phase)
    imag = lin * np.sin(phase)

    if plot_data:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
        axs[0].plot(freqs, powers_dB)
        axs[1].plot(freqs, phase)

    return (freqs, real, imag, powers_dB, phase)


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


def fit(freq, real, imag, mag, phase, real_only=0, bounds=None, f0Guess=None, QextGuess=None, QintGuess=None,
            plot=False, printout=False, n=3):

    if QextGuess == None:
        QextGuess = np.mean(freq)/(10*(freq[1]-freq[0]))
    if QintGuess == None:
        QintGuess = np.mean(freq)/(10*(freq[1]-freq[0]))

    S21 = real + 1j * imag

    if f0Guess == None:
        f0Guess = rough_guess(freq, S21, n=n)

    if bounds == None:
        bounds = ([QextGuess / 10.0, QintGuess / 10.0, np.min(freq)],
                  [QextGuess * 10.0, QintGuess * 10.0, np.max(freq)])

    target_func = transmission_func
    data_to_fit = np.abs(real + 1j * imag)

    popt, pcov = curve_fit(target_func, freq, data_to_fit,
                           p0=(QextGuess, QintGuess, f0Guess),
                           bounds=bounds,
                           maxfev=1e4, ftol=2.3e-8, xtol=2.3e-8)

    if printout:
        print(f'f (Hz): {rounder(popt[2] / 2 / np.pi)}', )
        fitting_params = list(inspect.signature(transmission_func).parameters.keys())[1:]
        for i in range(2):
            print(f'{fitting_params[i]}: {rounder(popt[i])} +- {rounder(np.sqrt(pcov[i, i]))}')
        Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
        print('Q_tot: ', rounder(Qtot), '\nT1 (s):', rounder(Qtot / popt[2]),
              f"Kappa: {rounder(popt[2] / 2 / np.pi / Qtot)}", )

        print(f"Kappa ext: {rounder(popt[2] / 2 / np.pi / popt[0])}",
              f"\nKappa int: {rounder(popt[2] / 2 / np.pi / popt[1])}", )

    if plot:
        popt_guess = [QextGuess, QintGuess, f0Guess]

        points_guess = plotRes(freq, real, imag, mag, phase, popt_guess)
        points_fit = plotRes(freq, real, imag, mag, phase, popt)

        plt.figure(figsize=(4,3))
        S21d = decimate_by_two(S21, n=n)
        metric = dx2(np.imag(S21d)) ** 2 + dx2(np.real(S21d)) ** 2
        plt.plot(metric)

    '''
    Currently, I need a way to estimate the fit confidence, but pcov is not a great way of doing this from 
    experience. Calculating a kind of residual instead.
    '''

    points_fit = transmission_func(freq, *popt) + 1j * transmission_func(freq, *popt)

    norm_fit = np.mean(np.abs(points_fit))
    cost_fit = np.mean(np.abs(points_fit - S21)**2) / norm_fit

    return popt, cost_fit

def fit_mode_from_ddh5(filepath, plot=False, printout=False, use_radians=True, freqs_name='frequency',
                       phase_name='phase', mag_name='power', n=3, f0Guess=None, QextGuess=None, QintGuess=None, real_only=False):

    (freq, real, imag, mag, phase) = getData_from_datadict(filepath, plot_data=plot, freqs_name=freqs_name, phase_name=phase_name,
                                 mag_name=mag_name, use_radians=use_radians)

    popt, pcov = fit(freq, real, imag, mag, phase, plot=plot, printout=printout, n=n, f0Guess=f0Guess, QextGuess=QextGuess,
                     QintGuess=QintGuess, real_only=real_only)

    return popt, pcov


def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq
    realRes = transmission_func(freq, *popt)
    imagRes = transmission_func(freq, *popt)
    # realRes = reflectionFunc(freq, *popt)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 1, 1)
    plt.title('real')
    plt.plot(xdata, np.abs(real+1j*imag), '.')
    plt.plot(xdata, np.abs(realRes+1j*imagRes))
    plt.show()

    return realRes + imagRes*1j

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

trace_file = r'X:\data\EmbeddedAmplifier\cooldown20260131\vna\cable_test\2026-02-05\vastaAl\2026-02-05T175908_4730e3dc-VNApower_0dBm_in30_outB\data.ddh5'
popt, pcov = fit_mode_from_ddh5(trace_file, use_radians=False, plot=True, n=5, QextGuess=1e5, QintGuess=1e3)

print(popt[0])
print(popt[1])
print(popt[2])

from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt
from Hatlab_DataProcessing.CWmode_fits.QFit import fit, plotRes

def get_fluxsweep_data_from_ddh5(filepath, trim_currents=(0,0), trim_freqs=(0,0)):
    '''
    trim is how many points *from the boundary* to delete
    '''

    FS_dd = all_datadicts_from_hdf5(filepath)['data']

    currents = np.unique(FS_dd['bias_current']['values'])
    freqs = np.unique(FS_dd['vna_frequency']['values'])
    phase = FS_dd['vna_phase']['values']
    mag = FS_dd['vna_power']['values']

    phase = phase.reshape((currents.shape[0], phase.shape[0] // currents.shape[0]))
    mag = mag.reshape((currents.shape[0], mag.shape[0] // currents.shape[0]))

    lin = 10 ** (mag / 20.0)

    imag = lin * np.sin(phase/180*np.pi)
    real = lin * np.cos(phase/180*np.pi)

    if trim_currents[0] > 0:
        currents = currents[trim_currents[0]:]
        real = real[trim_currents[0]:, :]
        imag = imag[trim_currents[0]:, :]
        phase = phase[trim_currents[0]:, :]
        mag = mag[trim_currents[0]:, :]

    if trim_currents[1] > 0:
        currents = currents[0:-trim_currents[1]]
        real = real[0:-trim_currents[1], :]
        imag = imag[0:-trim_currents[1], :]
        phase = phase[0:-trim_currents[1], :]
        mag = mag[0:-trim_currents[1], :]

    if trim_freqs[0] > 0:
        freqs = freqs[trim_freqs[0]:]
        real = real[:,trim_freqs[0]:]
        imag = imag[:,trim_freqs[0]:]
        phase = phase[:,trim_freqs[0]:]
        mag = mag[:,trim_freqs[0]:]

    if trim_freqs[1] > 0:
        freqs = freqs[0:-trim_freqs[1]]
        real = real[:,0:-trim_freqs[1]]
        imag = imag[:,0:-trim_freqs[1]]
        phase = phase[:,0:-trim_freqs[1]]
        mag = mag[:,0:-trim_freqs[1]]


    return freqs, currents, real, imag, mag, phase

def remove_fluxsweep_background(currents, real, imag, mag, phase):
    '''
    subtracts the part of the fluxsweep data that doesn't modulate with flux. Not very sophisticated, perhaps should
    improve this
    '''

    real_average = np.mean(real, axis=0)
    imag_average = np.mean(imag, axis=0)
    mag_average = np.mean(mag, axis=0)
    phase_average = np.mean(phase, axis=0)

    mag_average, b = np.meshgrid(mag_average, currents)
    magc = mag-mag_average - np.mean(mag)

    phase_average, b = np.meshgrid(phase_average, currents)
    phasec = phase-phase_average

    linc = 10 ** (magc / 20.0)

    imagc = linc * np.sin(phasec/180*np.pi)
    realc = linc * np.cos(phasec/180*np.pi)

    return realc, imagc, magc, phasec

def fit_fluxsweep(freqs, currents, real, imag, mag, phase, n=3):

    f0 = currents*0
    Qint = currents * 0
    Qext = currents * 0


    for i in range(0, len(currents)):

        real_i = real[i,:]
        imag_i = imag[i, :]
        mag_i = mag[i, :]
        phase_i = phase[i, :]

        popt, pcov = fit(freqs, real_i, imag_i, mag_i, phase_i, plot=False, n=n)

        Qext[i] = popt[0]
        Qint[i] = popt[1]
        f0[i] = popt[2]

    return Qext, Qint, f0

def fit_fluxsweep_from_ddh5(filepath, save_filepath=None, remove_background=False, trim_currents=(0, 0), trim_freqs=(0, 0), plot=False, n=3):

    # todo save fit to ddh5 file

    freqs, currents, real, imag, mag, phase = get_fluxsweep_data_from_ddh5(filepath, trim_currents=trim_currents, trim_freqs=trim_freqs)

    if remove_background:
        real, imag, mag, phase = remove_fluxsweep_background(currents, real, imag, mag, phase)

    Qext, Qint, f0 = fit_fluxsweep(freqs, currents, real, imag, mag, phase, n=n)

    if plot:
        plt.figure()
        plt.pcolor(currents, freqs, phase.transpose())
        plt.plot(currents, f0, 'r')
        plt.plot(currents, f0 - f0 / Qext, 'k--')
        plt.plot(currents, f0 + f0 / Qext, 'k--')
        plt.show()

    return currents, Qext, Qint, f0

def get_fluxsweep_fit_from_ddh5(filepath, current_name='currents', Qext_name='Qext', Qint_name='Qint', f0_name='f0'):
    
    datadict = all_datadicts_from_hdf5(filepath)['data']
    
    print(datadict)
    
    currents = datadict[current_name]['values']
    Qext = datadict[Qext_name]['values']
    Qint = datadict[Qint_name]['values']
    f0 = datadict[f0_name]['values']
    
    return currents, Qext, Qint, f0





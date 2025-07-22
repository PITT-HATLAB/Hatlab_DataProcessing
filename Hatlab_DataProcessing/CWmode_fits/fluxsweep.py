from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt
from Hatlab_DataProcessing.CWmode_fits.QFit import fit, plotRes
from plottr.data import datadict_storage as dds, datadict as dd
from tqdm import tqdm

def get_fluxsweep_data_from_ddh5(filepath, trim_currents=(0,0), trim_freqs=(0,0),
                                 currents_name='bias_current', freqs_name='vna_frequency', phase_name='vna_phase', mag_name='vna_power',use_radians=True):
    '''
    trim is how many points *from the boundary* to delete
    '''

    FS_dd = all_datadicts_from_hdf5(filepath)['data']

    currents = np.unique(FS_dd[currents_name]['values'])
    freqs = np.unique(FS_dd[freqs_name]['values'])
    phase = FS_dd[phase_name]['values']
    mag = FS_dd[mag_name]['values']

    phase = phase.reshape((currents.shape[0], phase.shape[0] // currents.shape[0]))
    mag = mag.reshape((currents.shape[0], mag.shape[0] // currents.shape[0]))

    if use_radians == False:
        phase *= np.pi/180

    lin = 10 ** (mag / 20.0)

    imag = lin * np.sin(phase)
    real = lin * np.cos(phase)

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
    cost = currents*0


    for i in tqdm(range(0, len(currents))):

        real_i = real[i,:]
        imag_i = imag[i, :]
        mag_i = mag[i, :]
        phase_i = phase[i, :]

        popt, cost_i = fit(freqs, real_i, imag_i, mag_i, phase_i, plot=False, n=n)

        Qext[i] = popt[0]
        Qint[i] = popt[1]
        f0[i] = popt[2]
        cost[i] = cost_i

    return Qext, Qint, f0, cost

def fit_fluxsweep_adaptive(freqs, currents, real, imag, mag, phase, f0Guess, QGuess, n=3):

    '''
    rather than running Qfit for each point individually, this fitter works from left to right, using the previous fit
    result as the guess for the next. It also uses a dynamic window to improve the fit
    '''

    f0 = currents*0
    Qint = currents * 0
    Qext = currents * 0

    QextGuess = None
    QintGuess = None
    AGuess = None
    BGuess = None
    CGuess = None
    DGuess = None

    for i in tqdm(range(0,len(currents))):

        f_interp = np.linspace(f0Guess - f0Guess/QGuess * 3, f0Guess + f0Guess/QGuess * 3, 1000)

        real_interp = np.interp(f_interp, freqs, real[i,:])
        imag_interp = np.interp(f_interp, freqs, imag[i, :])
        mag_interp = np.interp(f_interp, freqs, mag[i, :])
        phase_interp = np.interp(f_interp, freqs, phase[i, :])

        popt, cost = fit(f_interp, real_interp, imag_interp, mag_interp, phase_interp, f0Guess=f0Guess, AGuess=AGuess, BGuess=BGuess,
                             CGuess=CGuess, DGuess=DGuess, QintGuess=QintGuess, QextGuess=QextGuess, n=n, plot=False)

        Qext[i] = popt[0]
        Qint[i] = popt[1]
        f0[i] = popt[2]

        QextGuess = popt[0]
        QintGuess = popt[1]
        f0Guess = popt[2]
        AGuess = popt[3]
        BGuess = popt[4]
        CGuess = popt[5]
        DGuess = popt[6]

    return Qext, Qint, f0, cost


def fit_fluxsweep_from_ddh5(filepath, save_filepath=None, remove_background=False, trim_currents=(0, 0), trim_freqs=(0, 0), plot=False, n=3,use_radians=True, f0Guess=None, QGuess=None):

    # todo save fit to ddh5 file

    freqs, currents, real, imag, mag, phase = get_fluxsweep_data_from_ddh5(filepath, trim_currents=trim_currents, trim_freqs=trim_freqs, use_radians=use_radians)

    if remove_background:
        real, imag, mag, phase = remove_fluxsweep_background(currents, real, imag, mag, phase)

    if f0Guess != None and QGuess != None:
        print('Running adaptive fitter...')
        Qext, Qint, f0, cost = fit_fluxsweep_adaptive(freqs, currents, real, imag, mag, phase, f0Guess, QGuess, n=n)
    else:
        print('Running parallel fitter...')
        Qext, Qint, f0, cost = fit_fluxsweep(freqs, currents, real, imag, mag, phase, n=n)
    print('...done!')

    if plot:
        plt.figure()
        plt.pcolor(currents, freqs, phase.transpose())
        plt.plot(currents, f0, color=(0,0,0,0.3), linewidth=10)
        # plt.plot(currents, f0 - f0 / Qext, 'k--')
        # plt.plot(currents, f0 + f0 / Qext, 'k--')
        plt.show()

    if save_filepath != None:

        data = dd.DataDict(
            current=dict(unit='A'),
            fit_freq=dict(axes=['current'], unit='Hz'),
            fit_Qext=dict(axes=['current'], unit='Hz'),
            fit_Qint=dict(axes=['current'], unit='Hz'))

        data.validate()

        name = filepath.split('\\')[-2]

        with dds.DDH5Writer(datadict=data, basedir=save_filepath, name=name) as writer:

            writer.add_data(
                current=currents,
                fit_freq=f0,
                fit_Qext=Qext,
                fit_Qint=Qint)

    return currents, Qext, Qint, f0, cost

if __name__ == '__main__':
    from Hatlab_DataProcessing.CWmode_fits.fluxsweep import fit_fluxsweep_from_ddh5, get_fluxsweep_data_from_ddh5, \
        remove_fluxsweep_background
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    FSfile = r"X:\data\EmbeddedAmplifier\cooldown20250608\VNA\fluxsweep\2025-07-13\2025-07-13T002414_3bfbd726-in9_outC_mag3_-45Bm_VNA_VNA\data.ddh5"
    currents, Qext, Qint, f0, cost = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, trim_freqs=(0, 0), trim_currents=(0,50),
                                                       n=5, use_radians=True, f0Guess=3.925e9, QGuess=500, save_filepath=r'X:\data\EmbeddedAmplifier\cooldown20250608\fits')




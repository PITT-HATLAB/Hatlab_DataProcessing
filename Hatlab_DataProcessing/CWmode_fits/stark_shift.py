from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt
from Hatlab_DataProcessing.CWmode_fits.QFit import fit, plotRes
from plottr.data import datadict_storage as dds, datadict as dd
from tqdm import tqdm

def get_data_from_ddh5_adaptive_duffing(filepath, current_name='current', detuning_name='detuning', phase_name='phase',
                                 mag_name='power', f0_name='f0', use_radians=True):
    '''
    trim is how many points *from the boundary* to delete
    '''

    dd = all_datadicts_from_hdf5(filepath)['data']
    phase = dd[phase_name]['values'] * np.pi / 180
    mag = dd[mag_name]['values']
    drive_power = dd['drive_power']['values']
    current = dd[current_name]['values']
    detuning = np.unique(dd[detuning_name]['values'][0, :])
    f0sweep = dd[f0_name]['values']

    current_len = len(np.unique(current))
    drive_power_len = len(np.unique(drive_power))

    phase = np.reshape(phase, (current_len, drive_power_len, phase.size // current_len // drive_power_len))
    mag = np.reshape(mag, (current_len, drive_power_len, phase.size // current_len // drive_power_len))
    drive_power = np.reshape(drive_power, (current_len, drive_power_len))

    f0sweep = np.reshape(f0sweep, (current_len, drive_power_len))[:, 0]

    if use_radians == False:
        phase *= np.pi/180

    from Hatlab_DataProcessing.CWmode_fits.fluxsweep import fit_fluxsweep
    from Hatlab_DataProcessing.CWmode_fits.QFit import fit


def fit_stark_shift(currents, drive_powers, f0sweep, detuning):

    current_len = len(np.unique(currents))
    drive_power_len = len(np.unique(drive_powers))

    f0s = np.zeros([current_len, drive_power_len])
    kappa_ints = np.zeros([current_len, drive_power_len])
    kappa_exts = np.zeros([current_len, drive_power_len])

    for i in tqdm(range(0, current_len)):
        f0 = f0sweep[i]
        for j in range(0, drive_power_len):
            VNA_frequency = detuning + f0sweep[i]

            lin = 10 ** (mag[i, j, :] / 20.0)

            imag = lin * np.sin(phase[i, j, :])
            real = lin * np.cos(phase[i, j, :])

            freqs = VNA_frequency

            popt, cost = fit(freqs, real, imag, mag[i, j, :], phase[i, j, :], n=4, f0Guess=f0, QextGuess=100)

            f0 = popt[2]
            Qint = popt[1]
            Qext = popt[0]

            Qtot = 1 / (1 / Qext + 1 / Qint)

            kappa_ints[i, j] = f0 / Qint
            kappa_exts[i, j] = f0 / Qext

            f0s[i, j] = f0



    return f0, cost

def fit_stark_shift_from_ddh5(freqs, drive_power, real, imag, mag, phase, n=3):
    get_data_from_ddh5_adaptive_duffing(filepath, currents_name='bias_current', detuning_name='detuning',
                                        phase_name='phase',
                                        mag_name='power', f0_name='f0', use_radians=True)

    return f0, cost


if __name__ == '__main__':

    filepath = r'X:\data\EmbeddedAmplifier\cooldown20250608\VNA\duffing\2025-07-21\2025-07-21T231020_6699c066-Adaptive_Duffing\data.ddh5'

    detuning, currents, real, imag, mag, phase, f0 = get_data_from_ddh5_adaptive_duffing(filepath, currents_name='current')
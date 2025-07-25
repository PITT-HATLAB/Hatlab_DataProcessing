from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt
from Hatlab_DataProcessing.CWmode_fits.QFit import fit, plotRes
from plottr.data import datadict_storage as dds, datadict as dd
from tqdm import tqdm

def get_data_from_ddh5_adaptive_duffing(filepath, currents_name='bias_current', detuning_name='detuning', phase_name='phase',
                                 mag_name='power', f0_name='f0', use_radians=True):
    '''
    trim is how many points *from the boundary* to delete
    '''

    FS_dd = all_datadicts_from_hdf5(filepath)['data']

    currents = np.unique(FS_dd[currents_name]['values'])
    detuning = np.unique(np.round(FS_dd[detuning_name]['values']),-2)
    phase = FS_dd[phase_name]['values']
    mag = FS_dd[mag_name]['values']
    f0 = FS_dd[f0_name]['values']

    phase = phase.reshape((len(currents), len(detuning), phase.shape[0] // len(currents) // len(detuning)))
    mag = mag.reshape((len(currents), len(detuning), mag.shape[0] // len(currents) // len(detuning)))

    if use_radians == False:
        phase *= np.pi/180

    lin = 10 ** (mag / 20.0)

    imag = lin * np.sin(phase)
    real = lin * np.cos(phase)

    return detuning, currents, real, imag, mag, phase, f0


def fit_stark_shift(freqs, variable, real, imag, mag, phase, n=3):

    f0 = variable*0
    cost = variable*0

    for i in tqdm(range(0, len(variable))):

        real_i = real[i,:]
        imag_i = imag[i, :]
        mag_i = mag[i, :]
        phase_i = phase[i, :]

        popt, cost_i = fit(freqs, real_i, imag_i, mag_i, phase_i, plot=False, n=n)

        Qext[i] = popt[0]
        Qint[i] = popt[1]
        f0[i] = popt[2]
        cost[i] = cost_i

    return f0, cost


if __name__ == '__main__':

    filepath = r'X:\data\EmbeddedAmplifier\cooldown20250608\VNA\duffing\2025-07-21\2025-07-21T231020_6699c066-Adaptive_Duffing\data.ddh5'

    detuning, currents, real, imag, mag, phase, f0 = get_data_from_ddh5_adaptive_duffing(filepath, currents_name='current')
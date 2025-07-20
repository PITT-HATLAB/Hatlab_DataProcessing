import numpy as np
import matplotlib.pyplot as plt
import scipy

def snail_freq(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=3, M=1):
    '''
    Lj is total Lj
    '''

    omega, g3, g4, K = get_snail_hamiltonian_params(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=N, M=M)

    f0 = omega / (2 * np.pi)

    return np.array(f0, dtype=np.float64)

def get_snail_hamiltonian_params(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=3, M=1):

    '''
    Completely derived from Frattini et al. 2018
    '''

    def f(delta, phi_r, alpha, E_J):
        return -alpha * E_J * np.cos(delta) - N * E_J * np.cos((2 * np.pi * phi_r - delta) / N)

    Lj_single = (N*alpha+1)/ (N) * Lj # Lj of a single large-area junction. By Frattini convention, jj params are relative to this value

    hbar = 1.0545718e-34
    e = 1.60218e-19
    phi0 = np.pi * hbar / e
    Ic = phi0 / Lj_single
    E_J = Ic * phi0 / (2 * np.pi)

    phi_r = (Ib - I0) / dIdphi_r

    delta_s = phi_r * 0

    for i in range(0, len(phi_r)):
        res = scipy.optimize.minimize_scalar(f, args=(phi_r[i], alpha, E_J))
        delta_s[i] = res.x

    c2 = alpha * np.cos(delta_s) / 2 + np.cos(2 * np.pi * phi_r / N - delta_s / N) / (2 * N)
    c3 = -alpha * np.sin(delta_s) / 6 + np.sin(2 * np.pi * phi_r / N - delta_s / N) / 54
    c4 = -alpha * np.cos(delta_s) / 24 - np.cos(2 * np.pi * phi_r / N - delta_s / N) / 648

    Ls = Lj_single / c2
    omega = 1 / np.sqrt(C * (L + M * Ls))

    p = M * Ls / (L + M * Ls)

    E_C = 2 * e ** 2 / C

    g3 = 1 / 6 * p ** 2 / M * c3 / c2 * np.sqrt(E_C * hbar * omega) / hbar
    g4 = 1 / 12 * p ** 3 / M ** 2 * (c4 - 3 * c3 ** 2 / c2 * (1 - p)) * E_C / c2 / hbar

    K = 12 * (g4 - 5 * g3 ** 2 / omega)

    return omega, g3, g4, K


def fit_fluxsweep_to_snail_model(L, C, currents, f0s, Lj_guess, alpha_guess, dIdphi_r_guess, I0_guess, N=3, M=1,plot=False):
    '''
    Lj guess is total Lj of the SNAIL
    alpha is asymmetry parameter
    dIdpi_r is flux coupling constant
    I0 is static flux offset
    '''


    from scipy.optimize import curve_fit

    fit_func = lambda Ib, Lj, alpha, dIdphi_r, I0: snail_freq(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=N, M=M)

    p0 = [Lj_guess, alpha_guess, dIdphi_r_guess, I0_guess]

    popt, pcov = curve_fit(fit_func, currents, f0s, p0=p0, maxfev=100000, ftol=2.3e-16, xtol=2.3e-16)

    if plot:
        fit_f0 = fit_func(currents, popt[0], popt[1], popt[2], popt[3])
        guess_f0 = fit_func(currents, p0[0], p0[1], p0[2], p0[3])

        plt.figure()
        plt.plot(currents, guess_f0, label='guess')

        plt.plot(currents, fit_f0, label='fit')
        plt.scatter(currents, f0s, s=2, c='k', label='data')
        plt.xlabel('Current (A)')
        plt.ylabel('Freq. (Hz)')
        plt.legend()
        plt.show()

    return popt, pcov

if __name__ == '__main__':
    L = 4.161193488604595e-09
    C = 1.6100076629787958e-13

    dIdphi_r_guess = 4e-3
    I0_guess = -0.5e-3
    alpha_guess = 0.12
    Lj_guess = 1.5e-9

    from Hatlab_DataProcessing.CWmode_fits.fluxsweep import fit_fluxsweep_from_ddh5
    FSfile = r"W:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-24\2025-04-24T163529_a906f1c7-FS_pwr=-40dB_mag4_in10_outG_VNA_VNA\data.ddh5"
    currents, Qext, Qint, f0 = fit_fluxsweep_from_ddh5(FSfile, remove_background=True, plot=True, n=3, trim_freqs=(0,0), trim_currents=(0,50))

    popt, pcov = fit_fluxsweep_to_snail_model(L, C, currents, f0, Lj_guess, alpha_guess, dIdphi_r_guess, I0_guess, N=3, M=1, plot=True)

    phi_r = (currents - popt[3]) / popt[2]
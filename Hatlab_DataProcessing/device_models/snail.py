import numpy as np
import matplotlib.pyplot as plt
import scipy

def snail_freq(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=3, M=1):
    # a, Ej, delta_s, delta_ext, delta_min = symbols('alpha,E_j,delta_s,delta_ext, delta_min')

    '''
    Lj is total Lj
    '''

    def f(delta, phi_r, alpha, E_J):
        return -alpha * E_J * np.cos(delta) - N * E_J * np.cos((2 * np.pi * phi_r - delta) / N)

    Lj3 = (N*alpha+1)/(N)*Lj # Lj of a single large-area junction

    hbar = 1.0545718e-34
    e = 1.60218e-19
    phi0 = np.pi * hbar / e
    Ic = phi0 / Lj3
    phi_r = np.linspace(-0.5, 0.5, 7)
    E_J = Ic * phi0 / (2 * np.pi)

    phi_r = (Ib - I0) / dIdphi_r

    d_phi_r = phi_r[1] - phi_r[0]

    delta_s = phi_r * 0

    for i in range(0, len(phi_r)):
        res = scipy.optimize.minimize_scalar(f, args=(phi_r[i], alpha, E_J))
        delta_s[i] = res.x

    dphi_r = phi_r[1] - phi_r[0]

    ddelta_s = np.diff(delta_s) / d_phi_r

    slope = ddelta_s / dphi_r

    c2 = alpha * np.cos(delta_s) / 2 + np.cos(2 * np.pi * phi_r / N - delta_s / N) / (2 * N)

    f0 = (1 / np.sqrt(L * C)) / (np.sqrt(1 + Lj3 / (L * c2))) / (2 * np.pi)

    return np.array(f0, dtype=np.float64)

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
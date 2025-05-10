import numpy as np
import matplotlib.pyplot as plt
import scipy

def snail_freq(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=3, M=1):
    # a, Ej, delta_s, delta_ext, delta_min = symbols('alpha,E_j,delta_s,delta_ext, delta_min')

    def f(delta, phi_r, alpha, E_J):
        return -alpha * E_J * np.cos(delta) - N * E_J * np.cos((2 * np.pi * phi_r - delta) / N)

    hbar = 1.0545718e-34
    e = 1.60218e-19
    phi0 = np.pi * hbar / e
    Ic = phi0 / Lj
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

    f0 = (1 / np.sqrt(L * C)) / (np.sqrt(1 + Lj / (L * c2))) / (2 * np.pi)

    return np.array(f0, dtype=np.float64)

def fit_fluxsweep_to_snail_model(L, C, currents, f0s, Lj_guess, alpha_guess, dIdphi_r_guess, I0_guess, N=3, M=1, plot=False):
    '''
    Lj guess is total Lj of the SNAIL
    alpha is asymmetry parameter
    dIdpi_r is flux coupling constant
    I0 is static flux offset
    '''

    dIdphi_r = 4e-3
    I0 = 3.33e-3
    alpha_guess = 0.12
    Lj = 1.5e-9

    from scipy.optimize import curve_fit

    fit_func = lambda Ib, Lj, alpha, dIdphi_r, I0: snail_freq(Ib, Lj, alpha, L, C, dIdphi_r, I0, N=N, M=M)

    p0 = [Lj, alpha_guess, dIdphi_r, I0]

    popt, pcov = curve_fit(fit_func, currents, f0s, p0=p0)

    return popt, pcov


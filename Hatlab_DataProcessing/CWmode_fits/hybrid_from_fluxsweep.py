from Hatlab_DataProcessing.CWmode_fits.fluxsweep import fit_fluxsweep_from_ddh5
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Hatlab_DataProcessing.CWmode_fits.util import overlapping_support_points

def dressed_r_omega(omega_q, omega_r, g):
    '''
    Equation for the dressed eigenfrequency of a linear mode, as a function of some other mode, which presumable we
    can tune via flux biasing (or another method). One key approximation made is that the dressed frequency of the
    tuneable mode is roughly equal to its bare frequency, which should be true in dispersive limit

    See Blais et al. 2021, equation B30
    '''

    Delta = omega_r - omega_q

    if omega_r > np.mean(omega_q):
        omega_r_tilde = 0.5 * (omega_r + omega_q + np.sqrt(Delta ** 2 + 4 * g ** 2))

    else:
        omega_r_tilde = 0.5 * (omega_r + omega_q - np.sqrt(Delta ** 2 + 4 * g ** 2))

    return omega_r_tilde

def get_fluxquantum_from_FS(currents, f0, plot=False):

    def offset_sine(x, A, k, x0, y0):
        return A * np.cos((x - x0) * k) + y0

    Aguess = (np.max(f0) - np.min(f0)) / 2
    kGuess = 2 * np.pi / (np.max(currents) - np.min(currents))
    x0Guess = np.mean(currents)
    y0Guess = np.mean(f0)
    p0 = (Aguess, kGuess, x0Guess, y0Guess)
    bounds = ((Aguess / 1.1, kGuess / 3, np.min(currents), y0Guess / 1.1),
              (Aguess * 1.1, kGuess * 3, np.max(currents), y0Guess * 1.1))
    popt, pcov = curve_fit(offset_sine, currents, f0, p0=p0, bounds=bounds,maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)
    dphi_dIb = popt[1]
    zeroflux_current = popt[2]

    phi_r = (currents - zeroflux_current) * dphi_dIb

    if plot:

        plt.plot(currents, f0)
        plt.plot(currents, offset_sine(currents, *p0))
        plt.plot(currents, offset_sine(currents, *popt))

    return phi_r, dphi_dIb, zeroflux_current

def get_fluxquantum_from_FS_manual(currents, f0, zero_flux_current, half_flux_current, plot=False):

    dphi_dIb = np.pi/(half_flux_current-zero_flux_current)
    phi_r = (currents - zero_flux_current) * dphi_dIb

    return phi_r, dphi_dIb, zero_flux_current

def g_delta_fit(phi_r_s, phi_r_r, f0_s, f0_r, plot=False, printout=False):
    '''
    This is a fit for g/Delta, given flux sweep data for two modes. The _s means snail - i.e., the object with
    the superconducting loop. The _r means resonator - i.e., the thing with no flux modulation of the bare mode.

    It requires that the two fluxsweeps are defined over some overlapping range of flux phi_r - but the support
    points don't need to be the same.

    Use get_fluxquantum_from_FS(), if the flux sweeps don't have the same flux offset or flux coupling
    constant.
    '''

    phi_r = overlapping_support_points(phi_r_s, phi_r_r)

    f0_s_trimmed = np.interp(phi_r, phi_r_s, f0_s)
    f0_r_trimmed = np.interp(phi_r, phi_r_r, f0_r)

    f0_r_bare_guess = np.mean(f0_r_trimmed)

    f0_s_variation = np.max(f0_s_trimmed) - np.min(f0_s_trimmed)
    f0_r_variation = np.max(f0_r_trimmed) - np.min(f0_r_trimmed)

    delta_rough = np.abs(np.mean(f0_r_trimmed) - np.mean(f0_s_trimmed))

    gGuess = delta_rough*np.sqrt(f0_r_variation/f0_s_variation)

    bounds = ((f0_r_bare_guess/1.1*2*np.pi, gGuess/10.0*2*np.pi),(f0_r_bare_guess*1.1*2*np.pi, gGuess*10.0*2*np.pi))

    popt, pcov = curve_fit(dressed_r_omega, f0_s_trimmed*2*np.pi, f0_r_trimmed*2*np.pi,
                           p0=(f0_r_bare_guess*2*np.pi, gGuess*2*np.pi),
                           bounds=bounds,
                           maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)

    if plot:
        plt.figure()
        plt.scatter(f0_s_trimmed, f0_r_trimmed,label='data')
        plt.plot(f0_s_trimmed, dressed_r_omega(f0_s_trimmed*2*np.pi, popt[0], popt[1])/(2*np.pi),'k--',label='fit, g=' + str(np.round(popt[1]/(1e6*2*np.pi), 3)) + ' MHz')
        plt.plot(f0_s_trimmed, dressed_r_omega(f0_s_trimmed*2*np.pi, f0_r_bare_guess*2*np.pi,gGuess*2*np.pi)/(2*np.pi),'r--',label='Initial Guess')
        plt.legend()
        plt.show()

    g_delta = (popt[1]/(2*np.pi) / np.abs(f0_s_trimmed - f0_r_trimmed) )

    return phi_r, g_delta, popt, pcov

if __name__ == '__main__':
    # FSfile = r"Y:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-18T174406_40e5353d-FS_pwr=-55dB_mag3_in4_outC_VNA_VNA\data.ddh5"
    FSfile = r"W:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-24\2025-04-24T163529_a906f1c7-FS_pwr=-40dB_mag4_in10_outG_VNA_VNA\data.ddh5"
    currents_s, Qext_s, Qint_s, f0_s = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, trim_currents=(0,50))

    FSfile = r"W:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-26\2025-04-26T185655_c01889b4-in27_outF_mag4_-40dBm_VNA_VNA\data.ddh5"
    currents_r2, Qext_r2, Qint_r2, f0_r2 = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, n=7)
    f0_r2 = np.interp(currents_s, currents_r2, f0_r2)
    popt, pcov = fit(f0_s, f0_r2, plot=True)

    FSfile = r"W:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-26\2025-04-26T192046_15308a78-in10_outH_mag4_-45Bm_VNA_VNA\data.ddh5"
    currents_r1, Qext_r1, Qint_r1, f0_r1 = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, n=4, trim_freqs=(600,100))
    f0_r1 = np.interp(currents_s, currents_r1, f0_r1)
    popt, pcov = fit(f0_s, f0_r1, plot=True)

    FSfile = r"W:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-26\2025-04-26T190438_d4322af0-in7_outG_mag4_-65Bm_VNA_VNA\data.ddh5"
    currents_o, Qext_o, Qint_o, f0_o = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, n=5, trim_freqs=(200,0))
    f0_o = np.interp(currents_s, currents_o, f0_o)
    popt, pcov = fit(f0_s, f0_o, plot=True)

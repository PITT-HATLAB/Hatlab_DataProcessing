from Hatlab_DataProcessing.CWmode_fits.fluxsweep import fit_fluxsweep_from_ddh5
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


def fit(f0_s, f0_r, plot=False, printout=False):

    f0_r_bare_guess = np.mean(f0_r)

    f0_s_variation = np.max(f0_s)-np.min(f0_s)
    f0_r_variation = np.max(f0_r) - np.min(f0_r)

    delta_rough = np.abs(np.mean(f0_r) - np.mean(f0_s))

    gGuess = delta_rough*np.sqrt(f0_r_variation/f0_s_variation)

    bounds = ((f0_r_bare_guess/1.1*2*np.pi, gGuess/10.0*2*np.pi),(f0_r_bare_guess*1.1*2*np.pi, gGuess*10.0*2*np.pi))

    popt, pcov = curve_fit(dressed_r_omega, f0_s*2*np.pi, f0_r*2*np.pi,
                           p0=(f0_r_bare_guess*2*np.pi, gGuess*2*np.pi),
                           bounds=bounds,
                           maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)

    print(gGuess/1e6)

    if plot:
        plt.figure()
        plt.scatter(f0_s, f0_r,label='data')
        plt.plot(f0_s, dressed_r_omega(f0_s*2*np.pi, popt[0], popt[1])/(2*np.pi),'k--',label='fit, g=' + str(np.round(popt[1]/(1e6*2*np.pi), 3)) + ' MHz')
        plt.plot(f0_s, dressed_r_omega(f0_s*2*np.pi, f0_r_bare_guess*2*np.pi,gGuess*2*np.pi)/(2*np.pi),'r--',label='Initial Guess')
        plt.legend()
        plt.show()


    return popt, pcov

# FSfile = r"Y:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-18T174406_40e5353d-FS_pwr=-55dB_mag3_in4_outC_VNA_VNA\data.ddh5"
FSfile = r"Y:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-24\2025-04-24T163529_a906f1c7-FS_pwr=-40dB_mag4_in10_outG_VNA_VNA\data.ddh5"
currents_s, Qext_s, Qint_s, f0_s = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, trim_currents=(0,50))

# FSfile = r"Y:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-26\2025-04-26T185655_c01889b4-in27_outF_mag4_-40dBm_VNA_VNA\data.ddh5"
# currents_r2, Qext_r2, Qint_r2, f0_r2 = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, n=7)
# f0_r2 = np.interp(currents_s, currents_r2, f0_r2)
# popt, pcov = fit(f0_s, f0_r2, plot=True)

FSfile = r"Y:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-26\2025-04-26T192046_15308a78-in10_outH_mag4_-45Bm_VNA_VNA\data.ddh5"
currents_r1, Qext_r1, Qint_r1, f0_r1 = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, n=4, trim_freqs=(600,100))
f0_r1 = np.interp(currents_s, currents_r1, f0_r1)
popt, pcov = fit(f0_s, f0_r1, plot=True)

FSfile = r"Y:\data\EmbeddedAmplifier\cooldown20250403\VNA\fluxsweep\2025-04-26\2025-04-26T190438_d4322af0-in7_outG_mag4_-65Bm_VNA_VNA\data.ddh5"
currents_o, Qext_o, Qint_o, f0_o = fit_fluxsweep_from_ddh5(FSfile, remove_background=False, plot=True, n=5, trim_freqs=(200,0))
f0_o = np.interp(currents_s, currents_o, f0_o)
popt, pcov = fit(f0_s, f0_o, plot=True)


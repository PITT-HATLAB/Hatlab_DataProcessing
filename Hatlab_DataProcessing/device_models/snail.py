import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import scipy

init_printing(use_unicode = True)

a, Ej, delta_s, delta_ext, delta_min = symbols('alpha,E_j,delta_s,delta_ext, delta_min')


def f(delta, phi_r, alpha, E_J, N):
    return -alpha * E_J * np.cos(delta) - N * E_J * np.cos((2 * np.pi * phi_r - delta) / N)




delta = np.linspace(-10, 10, 201)

for i in range(0, len(phi_r)):
    plt.plot(delta, f(delta, phi_r[i], alpha, E_J, N), label=('Phi_r = %.1f' % phi_r[i]))

plt.xlabel('$\\delta_s$')
plt.ylabel('$U$')
plt.title('Energy of SNAIL')
plt.legend()
plt.grid()

Ljtotal = 1 / (1 / (Ljbig * N) + 1 / (Ljbig / alpha))
print(Ljtotal)
print(Ljbig / alpha)


class SNAIL_mode():
    def __init__(self, N, Ljs,  L, C, M=1):
        '''
        N: number of large area junctions, typically 2 or 3
        Ljs: list of length two, large area junction inductance and small area junction inductance (order doesn't matter, it will be handled)
        M: number of arrayed SNAILs, defaults to 1
        L: linear inductance (H) of mode
        C: capacitance (F) of mode
        '''

        hbar = 1.0545718e-34
        e = 1.60218e-19
        phi0 = np.pi * hbar / e
        Ljbig = 2.0e-9  # inductance of 1 big junction
        alpha = 0.05
        phi_r = np.linspace(-1, 1, 7)
        E_J = Ic * phi0 / (2 * np.pi)
        N = 3

        phi_r = np.linspace(-1, 1, 20001)

        d_phi_r = phi_r[1] - phi_r[0]

        delta_s = phi_r * 0

        for i in range(0, len(phi_r)):
            res = scipy.optimize.minimize_scalar(f, args=(phi_r[i], alpha, E_J, N))
            delta_s[i] = res.x

        dphi_r = phi_r[1] - phi_r[0]

        ddelta_s = np.diff(delta_s) / d_phi_r

        slope = ddelta_s / dphi_r

        plt.plot(phi_r, delta_s)
        plt.xlabel('$\phi/\phi_0$')
        plt.ylabel('$\delta_s$')
        plt.title('$\delta_s$ at energy minimization condition')
        plt.grid()

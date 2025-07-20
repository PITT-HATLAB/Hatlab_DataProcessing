'''
Author: Boris Mesits
Much of this derives from Ryan Kaufman's code from 2021.

Some issues with this code I have:
- it doesn't separate the snail element itself (coeffs, Ej) from the amp mode (L, C, kappa, etc)
- it is formatted weird, it has object attributes but doesn't use them
- doesn't generalize to 2-SNAIL
- found a typo, where total Lj of snail is calculated wrong (assumes only 1 JJ on each arm)
- incorrect kerr null point
- slow
'''

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from timeit import default_timer as timer


def parallel(v1, v2):
    return 1 / (1 / v1 + 1 / v2)

def get_phi_min_funcs(alpha, phi_ext_arr):
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail_norm = -a * sp.cos(phi_s) - 3 * sp.cos((phi_e - phi_s) / 3)
    c1 = sp.series(U_snail_norm, phi_s, x0=phi_m, n=2).removeO().coeff((phi_s - phi_m))
    # generate a lambda function that outputs another lambda function for a given phi_ext
    # which then depends on phi_m only
    func_arr = []
    for phi_ext in phi_ext_arr:
        c1_num = sp.lambdify(phi_m, c1.subs(a, alpha).subs(phi_e, phi_ext), "numpy")
        func_arr.append(c1_num)
    return func_arr


def get_phi_min_fsolve(alpha, phi_ext_arr):
    funcs = get_phi_min_funcs(alpha, phi_ext_arr)
    sol_arr = np.ones(np.size(funcs))
    for i, func in enumerate(funcs):
        sol_arr[i] = fsolve(func, phi_ext_arr[i])
    return sol_arr


def get_phi_min(alpha, phi_ext):
    func = get_phi_min_funcs(alpha, [phi_ext])[0]
    return (fsolve(func, phi_ext)[0])


def c4_func_gen_vectorize(alpha_val):  # can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a * sp.cos(phi_s) - 3 * sp.cos((phi_e - phi_s) / 3))
    expansion = sp.series(U_snail, phi_s, x0=phi_m, n=5)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s - phi_m, 4)) * 24
    c4exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c4exp)


def c3_func_gen_vectorize(alpha_val):  # can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a * sp.cos(phi_s) - 3 * sp.cos((phi_e - phi_s) / 3))
    expansion = sp.series(U_snail, phi_s, x0=phi_m, n=4)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s - phi_m, 3)) * 6
    c3exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c3exp)


def c2_func_gen_vectorize(alpha_val):
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a * sp.cos(phi_s) - 3 * sp.cos((phi_e - phi_s) / 3))
    expansion = sp.series(U_snail, phi_s, x0=phi_m, n=3)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s - phi_m, 2)) * 2
    c2exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c2exp)


class SnailElement():
    def __init__(self):  # uA/um^2
        '''
        Parameters
        ----------
        junction_sizes : tuple
            (small_size, large_size) in micrometers squared
        quanta_start : float
            0-flux point in Amps
        quanta_size : float
            quanta ize in Amps

        Returns
        -------
        None.
        '''

        self.hbar = 1.0545718e-34
        self.e = 1.60218e-19
        self.phi0 = 2 * np.pi * self.hbar / (2 * self.e)

    def generate_quanta_function(self, quanta_offset, quanta_size):
        # function for converting bias currents to quanta fractions
        self.quanta_offset = quanta_offset
        self.quanta_size = quanta_size
        self.conv_func = lambda c: (c - quanta_offset) / quanta_size

    def info_from_junction_sizes(self, junction_sizes, res=100, Jc=0.8, verbose=False):

        self.s_size, self.l_size = junction_sizes

        self.alpha_from_sizes = self.s_size / self.l_size
        self.I0s, self.I0l = Jc * self.s_size * 1e-6, Jc * self.l_size * 1e-6

        self.Lss, self.Lsl = self.Ic_to_Lj(self.I0s), self.Ic_to_Lj(self.I0l)
        self.Ejs, self.Ejl = self.Ic_to_Ej(self.I0s), self.Ic_to_Ej(self.I0l)

        self.Ls0 = parallel(self.Lss, self.Lsl)

        self.c2_func, self.c3_func, self.c4_func = self.generate_coefficient_functions(self.alpha_from_sizes, res=res,
                                                                                       verbose=verbose)

        return self.c2_func, self.c3_func, self.c4_func

    def info_from_junction_i0(self, junction_i0_small, junction_i0_large, res=100, Jc=0.8, verbose=False):
        '''
        junction_i0_small: junction critical current in A
        junction_i0_large: junction critical current in A
        '''

        self.I0s, self.I0l = junction_i0_small, junction_i0_large

        self.Lss, self.Lsl = self.Ic_to_Lj(self.I0s), self.Ic_to_Lj(self.I0l)
        self.Ejs, self.Ejl = self.Ic_to_Ej(self.I0s), self.Ic_to_Ej(self.I0l)

        self.alpha_from_i0 = self.Ejs / self.Ejl

        self.c2_func, self.c3_func, self.c4_func = self.generate_coefficient_functions(self.alpha_from_i0, res=res,
                                                                                       verbose=verbose)

        return self.c2_func, self.c3_func, self.c4_func

    def Ic_to_Ej(self, Ic: float):
        '''
        Parameters
        ----------
        Ic : float
            critical current in amps
        Returns
        -------
        Ej in Joules
        src: https://en.wikipedia.org/wiki/Josephson_effect
        '''
        return Ic * self.phi0 / (2 * np.pi)

    def Ic_to_Lj(self, Ic: float):
        '''
        Parameters
        ----------
        Ic : float
            critical current in amps
        Returns
        -------
        Lj in Henries
        src: https://en.wikipedia.org/wiki/Josephson_effect
        '''
        return self.phi0 / (2 * np.pi * Ic)

    def generate_coefficient_functions(self, alpha_val, res=int(100), plot=False, show_coefficients=False,
                                       verbose=False):
        '''
        Parameters
        ----------
        alpha_val : float
            alpha value between 0 and 0.33
        res : int, optional
            number of points to base interpolation off of. The default is 100.
        Returns
        -------
        c2_func : lambda function
            function that will return the value of c2
        c3_func : lambda function
            DESCRIPTION.
        c4_func : lambda function
            DESCRIPTION.

        '''
        if verbose:
            print("Calculating expansion coefficients")
        start_time = timer()

        #c4
        phi_ext_arr = np.linspace(0, 2 * np.pi, res)
        c4_arr = c4_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        if verbose:
            print(f"Elapsed time: {np.round(end_time - start_time, 2)} seconds")
        c4_func = interp1d(phi_ext_arr, c4_arr, 'quadratic')

        # c3:
        start_time = timer()
        phi_ext_arr = np.linspace(0, 2 * np.pi, res)
        c3_arr = c3_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        if verbose:
            print(f"Elapsed time: {np.round(end_time - start_time, 2)} seconds")
        c3_func = interp1d(phi_ext_arr, c3_arr, 'quadratic')

        # c2:
        start_time = timer()
        phi_ext_arr = np.linspace(0, 2 * np.pi, res)
        c2_arr = c2_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        if verbose:
            print(f"Elapsed time: {np.round(end_time - start_time, 2)} seconds")
        c2_func = interp1d(phi_ext_arr, c2_arr, 'quadratic')

        if plot:
            plt.plot(phi_ext_arr, self.c2_func(phi_ext_arr), label="c2")
            plt.plot(phi_ext_arr, self.c3_func(phi_ext_arr), label="c3")
            plt.plot(phi_ext_arr, self.c4_func(phi_ext_arr), label='c4')

            plt.legend()

        return c2_func, c3_func, c4_func

    def set_linear_inductance(self, L0):
        self.L0 = L0

    def set_linear_capacitance(self, C0):
        self.C0 = C0

    def generate_participation_function(self, L0, Lfunc):
        return lambda phi: Lfunc(phi) / (L0 + Lfunc(phi))

    def generate_inductance_function(self, L_large, c2_func):
        return lambda phi: L_large / c2_func(phi)

    def generate_resonance_function_via_LC(self, L0, C0, Ls_func):
        return lambda phi: 1 / np.sqrt((L0 + Ls_func(phi)) * C0)
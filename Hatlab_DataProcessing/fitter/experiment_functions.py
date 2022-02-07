from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import lmfit
from fitter_base import Fit, FitResult


class PiPulseTuneUp(Fit):
    @staticmethod
    def model(coordinates, amp, tau):
        """ amp * exp(-1.0 * x / tau)"""
        return amp * np.exp(-1.0 * coordinates / tau)
    @staticmethod
    def guess(coordinates, data):
        return dict(amp=1, tau=2)


    with open(yamlFile) as file:
        yamlDict = yaml.load(file, Loader=yaml.FullLoader)
    if xdata is None:
        piPulseAmpInfo = yamlDict['regularMsmtPulseInfo']['piPulseTuneUpAmp']
        xdata = np.linspace(piPulseAmpInfo[0], piPulseAmpInfo[1], piPulseAmpInfo[2] + 1)[:100]
    deriv = []
    for i in range(2001):
        angle = 0.001 * i
        iq_temp = rotate_complex(i_data, q_data, angle)
        yvalue = iq_temp.imag
        line_fit = np.zeros(len(yvalue)) + yvalue.mean()
        deriv_temp = ((yvalue - line_fit) ** 2).sum()
        deriv.append(deriv_temp)
    final = 0.001 * np.argwhere(np.array(deriv) == np.min(np.array(deriv)))
    rotation_angle = final.ravel()[0]
    print('The rotation angle is', rotation_angle, 'pi')

    iq_new = rotate_complex(i_data, q_data, rotation_angle)
    out = cos_fit(xdata, iq_new.real, plot=plot)
    freq = out.params.valuesdict()['freq']
    period = 1.0 / freq
    pi_pulse_amp = period / 2.0
    print('Pi pulse amp is ', pi_pulse_amp, 'V')
    fit_result = cos_model(out.params, xdata)
    excited_b, ground_b = determine_ge_states(xdata, fit_result)
    store_rot_info(rotation_angle, excited_b, ground_b, pi_pulse_amp)
    if plot:
        plt.plot(xdata, iq_new.imag)
        hline()
    if updatePiPusle_amp==1:
        with open(yamlFile) as file:
            info = yaml.load(file, Loader=yaml.FullLoader)
        info['pulseParams']['piPulse_gau']['amp'] = float(np.round(pi_pulse_amp, 4))
        with open(yamlFile, 'w') as file:
            yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    elif updatePiPusle_amp==2:
        if float(np.round(pi_pulse_amp, 4)) < 1:
            with open(yamlFile) as file:
                info = yaml.load(file, Loader=yaml.FullLoader)
            info['pulseParams']['piPulse_gau']['amp'] = float(np.round(pi_pulse_amp, 4))
            with open(yamlFile, 'w') as file:
                yaml.safe_dump(info, file, sort_keys=0, default_flow_style=None)
    else:
        pass
    return pi_pulse_amp


class T1Decay(Fit):
    @staticmethod
    def model(coordinates, amp, tau):
        """ amp * exp(-1.0 * x / tau)"""
        return amp * np.exp(-1.0 * coordinates / tau)
    @staticmethod
    def guess(coordinates, data):
        return dict(amp=1, tau=2)


class T2Ramsey(Fit):
    @staticmethod
    def model(coordinates, amp, tau, freq, phase):
        """ amp * exp(-1.0 * x / tau) * sin(2 * PI * freq * x + phase) """
        return amp * np.exp(-1.0 * coordinates / tau) * \
               np.sin(2 * np.pi * freq * coordinates + phase)
    @staticmethod
    def guess(coordinates, data):
        return dict(amp=1, tau=2, freq=3, phase=4)



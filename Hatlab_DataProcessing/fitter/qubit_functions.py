from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import lmfit
from matplotlib import pyplot as plt
from Hatlab_DataProcessing.fitter.fitter_base import Fit, FitResult
from Hatlab_DataProcessing.fitter.generic_functions import Cosine, ExponentialDecay, ExponentialDecayWithCosine
from Hatlab_DataProcessing.base import Analysis, AnalysisResult
from Hatlab_DataProcessing.helpers.unit_converter import t2f


class QubitBasicResult(AnalysisResult):
    def __init__(self, lmfit_result, parameters: Dict[str, Union[Dict[str, Any], Any]]):
        super().__init__(parameters)
        self.lmfit_result=lmfit_result

    def plot(self, figName=None):
        x_data = self.lmfit_result.userkws["coordinates"]
        result_str = self.params["result_str"].value
        plt.figure(figName)
        plt.title(result_str)
        plt.plot(x_data, self.lmfit_result.data, "o")
        plt.plot(x_data, self.lmfit_result.best_fit)

class PiPulseTuneUp(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, of) -> np.ndarray:
        """$A \cos(2 \pi f x + \phi) + of$"""
        return Cosine.model(coordinates, A, f, phi, of)

    @staticmethod
    def guess(coordinates, data):
        return Cosine.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, **fit_kwargs) -> QubitBasicResult:
        cosFitResult = super().analyze(coordinates, data, dry=False, params={}, **fit_kwargs)

        fit_phi = cosFitResult.params["phi"].value
        fit_f = cosFitResult.params.valuesdict()['f']

        zero_amp, pi_2_pulse_amp, pi_pulse_amp = \
            np.array(sorted([np.pi - fit_phi, np.pi / 2 - fit_phi, -fit_phi], key=lambda x: abs(x))) / 2 / np.pi / fit_f

        result_str = f'Pi pulse amp:{str(pi_pulse_amp)[:8]} DAC, Pi/2 pulse amp:{str(pi_2_pulse_amp)[:8]} DAC'
        print(result_str)

        return QubitBasicResult(cosFitResult,
                                dict(zero_amp=zero_amp,pi_pulse_amp=pi_pulse_amp, pi_2_pulse_amp=pi_2_pulse_amp,
                                     result_str=result_str))


class T1Decay(Fit):
    @staticmethod
    def model(coordinates, A, tau, of):
        """ A * exp(-1.0 * x / tau) + of"""
        return ExponentialDecay.model(coordinates, A, tau, of)

    @staticmethod
    def guess(coordinates, data):
        return ExponentialDecay.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, time_unit="us", **fit_kwargs):
        fitResult = super().analyze(coordinates, data, dry=False, params={}, **fit_kwargs)

        fit_tau = fitResult.params["tau"].value
        result_str = f'tau is {str(fit_tau)[:5]} {time_unit}'
        print(result_str)

        return QubitBasicResult(fitResult, dict(tau=fit_tau, result_str=result_str))
#
#
class T2Ramsey(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, tau, of):
        """ A * cos(2 pi f x + phi) * exp (-x/tau) + of"""
        return ExponentialDecayWithCosine.model(coordinates, A, f, phi, tau, of)

    @staticmethod
    def guess(coordinates, data):
        return ExponentialDecayWithCosine.guess(coordinates, data)

    def analyze(self, coordinates, data, dry=False, params={}, time_unit="us", **fit_kwargs):
        fitResult = super().analyze(coordinates, data, dry=False, params={}, **fit_kwargs)

        fit_tau = fitResult.params["tau"].value
        fit_f = fitResult.params["f"].value
        result_str = f'tau is {str(fit_tau)[:5]} {time_unit}, detuning is {f"{fit_f:.6e}"} {t2f(time_unit)}'
        print(result_str)

        return QubitBasicResult(fitResult, dict(tau=fit_tau, f=fit_f, result_str=result_str))


if __name__ == "__main__":
    x_data = np.linspace(-30000, 30000, int(1e6+1))
    y_data = -np.cos(2*np.pi/20000*x_data + 0.001) + 0.2

    piPulseFit = PiPulseTuneUp(x_data, y_data)
    # cosFitResult = cosFit.run(params={"A":lmfit.Parameter("A", 1, vary=False)}) # example of adjusting fitting parameter
    fitResult = piPulseFit.run()
    fitResult.plot()



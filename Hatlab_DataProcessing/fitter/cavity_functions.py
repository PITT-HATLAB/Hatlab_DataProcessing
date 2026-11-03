from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
from Hatlab_DataProcessing.fitter.fitter_base import Fit, FitResult
from Hatlab_DataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase

TWOPI = 2 * np.pi
PI = np.pi


def getVNAData(filename, freq_unit='Hz', plot=1, trim=0):
    trim_end = None if trim == 0 else -trim
    f = h5py.File(filename, 'r')
    freq = f['Freq'][()][trim: trim_end] * freqUnit(freq_unit)
    phase = f['S21'][()][1][trim: trim_end] / 180 * np.pi
    mag = f['S21'][()][0][trim: trim_end]
    f.close()

    lin = 10 ** (mag / 20.0)
    real = lin * np.cos(phase)
    imag = lin * np.sin(phase)

    if plot:
        plt.figure('mag')
        plt.plot(freq / 2 / np.pi, mag)
        plt.figure('phase')
        plt.plot(freq / 2 / np.pi, phase)

    return (freq, real, imag, mag, phase)


def cav_ref_func(freq, Qext, Qint, f0):
    """"reflection function of a harmonic oscillator"""
    omega0 = f0 * TWOPI
    delta = freq * TWOPI - omega0
    S_11_nume = 1 - Qint / Qext - 1j * 2 * Qint * delta / omega0
    S_11_denom = 1 + Qint / Qext - 1j * 2 * Qint * delta / omega0
    S11 = (S_11_nume / S_11_denom)
    return S11


class CavReflectionResult():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.f0 = self.params["f0"].value
        self.Qext = self.params["Qext"].value
        self.Qint = self.params["Qint"].value
        self.Qtot = self.Qext * self.Qint / (self.Qext + self.Qint)
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, **figArgs):
        real_fit = self.lmfit_result.best_fit.real
        imag_fit = self.lmfit_result.best_fit.imag
        mag_fit, phase_fit = realImag2magPhase(real_fit, imag_fit)
        mag_data, phase_data = realImag2magPhase(self.lmfit_result.data.real,
                                                 self.lmfit_result.data.imag)

        fig_args_ = dict(figsize=(12, 5))
        fig_args_.update(figArgs)
        plt.figure(**fig_args_)
        plt.subplot(1, 2, 1)
        plt.title('mag (dB pwr)')
        plt.plot(self.freqData, mag_data, '.')
        plt.plot(self.freqData, mag_fit)
        plt.subplot(1, 2, 2)
        plt.title('phase')
        plt.plot(self.freqData, phase_data, '.')
        plt.plot(self.freqData, phase_fit)
        plt.show()

    def print(self):
        print(f'f (Hz): {rounder(self.f0, 9)}+-{rounder(self.params["f0"].stderr, 9)}')
        print(f'Qext: {rounder(self.Qext, 5)}+-{rounder(self.params["Qext"].stderr, 5)}')
        print(f'Qint: {rounder(self.Qint, 5)}+-{rounder(self.params["Qint"].stderr, 5)}')
        print('Q_tot: ', rounder(self.Qtot, 5))
        print('T1 (s):', rounder(self.Qtot / self.f0 / 2 / np.pi, 5), '\nMaxT1 (s):',
              rounder(self.Qint / self.f0 / 2 / np.pi, 5))
        print('kappa/2Pi: ', rounder(self.f0 / self.Qtot / 1e6), 'MHz')


class CavReflection(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, conjugate:bool=True):
        """ fit cavity reflection function
        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.conjugate = conjugate
        self.pre_process()

    def model(self, coordinates, Qext, Qint, f0, magBack, phaseOff) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        S11 = cav_ref_func(coordinates, Qext, Qint, f0)
        if self.conjugate:
            S11 = S11.conjugate()
        S11 *= magBack * np.exp(1j * phaseOff)
        return S11

    @staticmethod
    def guess(coordinates, data):
        freq = coordinates
        phase = np.unwrap(np.angle(data))
        amp = np.abs(data)

        f0Guess = freq[np.argmin(amp)]  # smart guess of "it's probably the lowest point"
        magBackGuess = np.average(amp[:int(len(freq) / 5)])
        phaseOffGuess = phase[0]

        # guess algorithm from https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
        Q_min = 0.1 * (f0Guess / (freq[-1] - freq[0]))  # assume the user isn't trying to fit just a small part of a resonance curve
        delta_f = np.diff(freq)  # assume f is sorted
        min_delta_f = delta_f[delta_f > 0].min()
        Q_max = f0Guess / min_delta_f  # assume data actually samples the resonance reasonably
        QtotGuess = np.sqrt(Q_min * Q_max)  # geometric mean, why not?
        QextGuess = 2 * QtotGuess / (1 - amp[np.argmin(amp)]/magBackGuess)
        QintGuess = 1 / (1 / QtotGuess - 1 / QextGuess)

        Qext = lmfit.Parameter("Qext", value=QextGuess, min=QextGuess / 100, max=QextGuess * 100)
        Qint = lmfit.Parameter("Qint", value=QintGuess, min=QintGuess / 100, max=QintGuess * 100)
        f0 = lmfit.Parameter("f0", value=f0Guess, min=freq[0], max=freq[-1])
        magBack = lmfit.Parameter("magBack", value=magBackGuess, min=magBackGuess / 1.1, max=magBackGuess * 1.1)
        phaseOff = lmfit.Parameter("phaseOff", value=phaseOffGuess, min=-TWOPI, max=TWOPI)

        return dict(Qext=Qext, Qint=Qint, f0=f0, magBack=magBack, phaseOff=phaseOff)

    def run(self, *args: Any, **kwargs: Any) -> CavReflectionResult:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return CavReflectionResult(lmfit_result)



class CavReflectionResult_Phase():
    def __init__(self, lmfit_result: lmfit.model.ModelResult):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.eDelay = self.params["eDelay"].value
        self.f0 = self.params["f0"].value
        self.Qext = self.params["Qext"].value
        self.Qint = self.params["Qint"].value
        self.Qtot = self.Qext * self.Qint / (self.Qext + self.Qint)
        self.freqData = lmfit_result.userkws[lmfit_result.model.independent_vars[0]]

    def plot(self, **figArgs):
        phase_fit = self.lmfit_result.best_fit
        phase_data = self.lmfit_result.data

        fig_args_ = dict(figsize=(7, 5))
        fig_args_.update(figArgs)
        plt.figure(**fig_args_)
        plt.title('phase')
        plt.plot(self.freqData, phase_data, '.')
        plt.plot(self.freqData, phase_fit)
        plt.show()

    def print(self):
        print(f'f (Hz): {rounder(self.f0, 9)}+-{rounder(self.params["f0"].stderr, 9)}')
        print(f'Qext: {rounder(self.Qext, 5)}+-{rounder(self.params["Qext"].stderr, 5)}')
        print(f'Qint: {rounder(self.Qint, 5)}+-{rounder(self.params["Qint"].stderr, 5)}')
        print('Q_tot: ', rounder(self.Qtot, 5))
        print('T1 (s):', rounder(self.Qtot / self.f0 / 2 / np.pi, 5), '\nMaxT1 (s):',
              rounder(self.Qint / self.f0 / 2 / np.pi, 5))
        print('kappa/2Pi: ', rounder(self.f0 / self.Qtot / 1e6), 'MHz')


class CavReflectionPhaseOnly(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, conjugate:bool=True):
        """ fit cavity reflection function phase only.

        :param conjugate: fit to conjugated cavity reflection function (for VNA data)
        """
        self.coordinates = coordinates
        self.data = data
        self.conjugate = conjugate
        self.pre_process()

    def pre_process(self):
        self.data = np.unwrap(np.angle(self.data))

    def model(self, coordinates, Qext, Qint, f0, phaseOff, eDelay) -> np.ndarray:
        """"reflection function of a harmonic oscillator"""
        S11 = cav_ref_func(coordinates, Qext, Qint, f0)
        if self.conjugate:
            S11 = S11.conjugate()
        S11 *= np.exp(1j * (phaseOff + eDelay * (coordinates - f0) * TWOPI))
        phase = np.unwrap(np.angle(S11))
        return phase

    @staticmethod
    def guess(coordinates, data):
        freq = coordinates
        phase = data

        # f0_idx = int(np.floor(np.average(np.where(abs(phase - np.average(phase)) < 0.2))))
        f0_idx = len(phase)//2
        f0Guess = freq[f0_idx]
        phaseOffGuess = np.mean(phase)
        eDelayGuess = (phase[f0_idx//3] - phase[0]) / (freq[f0_idx//3] - freq[0]) / TWOPI

        # guess algorithm from https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html
        Q_min = 0.1 * (f0Guess / (freq[-1] - freq[0]))  # assume the user isn't trying to fit just a small part of a resonance curve
        delta_f = np.diff(freq)  # assume f is sorted
        min_delta_f = delta_f[delta_f > 0].min()
        Q_max = f0Guess / min_delta_f  # assume data actually samples the resonance reasonably
        QtotGuess = np.sqrt(Q_min * Q_max)  # geometric mean, why not?
        QextGuess = QtotGuess * 2
        QintGuess = 1 / (1 / QtotGuess - 1 / QextGuess)

        Qext = lmfit.Parameter("Qext", value=QextGuess, min=QextGuess / 100, max=QextGuess * 100)
        Qint = lmfit.Parameter("Qint", value=QintGuess, min=QintGuess / 100, max=QintGuess * 100)
        f0 = lmfit.Parameter("f0", value=f0Guess, min=freq[0], max=freq[-1])
        phaseOff = lmfit.Parameter("phaseOff", value=phaseOffGuess, min=-TWOPI*4, max=TWOPI*4)
        eDelay = lmfit.Parameter("eDelay", value=eDelayGuess)

        return dict(Qext=Qext, Qint=Qint, f0=f0, phaseOff=phaseOff, eDelay=eDelay)

    def run(self, *args: Any, **kwargs: Any) -> CavReflectionResult_Phase:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return CavReflectionResult_Phase(lmfit_result)




def cav_trans_func(freq, QL, f0):
    """Unit-height two-port transmission peak; freq and f0 are in Hz.

    QL is loaded Q, including internal loss and loading from BOTH ports.
    A free complex amplitude absorbs measurement gain and port imbalance,
    so this model does not independently determine Qint or either port Q.
    """
    return 1.0 / (1.0 - 2j * QL * (     np.asarray(freq) - f0) / f0)


# Compatibility with the previous peak function name.
cav_trans_peak_func = cav_trans_func


class CavTransmissionResult:
    """Complex S21 fit result, retaining .params, .plot() and .print()."""
    def __init__(self, lmfit_result):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.f0 = self.params['f0'].value
        self.freqData = lmfit_result.userkws['coordinates']
        self.Qtot = self.params['QL'].value
        self.QL = self.Qtot
        self.eDelay = self.params['eDelay'].value

    def plot(self, **figArgs):
        options = dict(figsize=(15, 4))
        options.update(figArgs)
        fig, axes = plt.subplots(1, 3, **options)
        data, fit = self.lmfit_result.data, self.lmfit_result.best_fit
        for values, label, style in [(data, 'data', '.'), (fit, 'fit', '-')]:
            axes[0].plot(self.freqData, 20*np.log10(np.maximum(np.abs(values),
                         np.finfo(float).tiny)), style, label=label)
            phase = np.unwrap(np.angle(values))
            # Align whole-turn branches for display only; fit complex data.
            if label == 'fit':
                phase += TWOPI * np.round(np.median(
                    (np.unwrap(np.angle(data)) - phase) / TWOPI))
            axes[1].plot(self.freqData, phase, style, label=label)
            axes[2].plot(values.real, values.imag, style, label=label)
        axes[0].set(xlabel='Frequency (Hz)', ylabel='|S21| (dB)')
        axes[1].set(xlabel='Frequency (Hz)', ylabel='Phase (rad)')
        axes[2].set(xlabel='Re(S21)', ylabel='Im(S21)')
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        plt.show()
        return fig, axes

    def print(self):
        for name in ('f0', 'QL', 'magBack', 'phaseOff', 'eDelay'):
            if name in self.params:
                p = self.params[name]
                uncertainty = 'unavailable' if p.stderr is None else f'{p.stderr:.6g}'
                print(f'{name}: {p.value:.8g} +/- {uncertainty}')
        print(f'Q_tot: {self.Qtot:.8g}')
        print(f'T1 loaded (s): {self.Qtot / (TWOPI*self.f0):.6g}')
        print(f'kappa/2Pi: {self.f0 / self.Qtot / 1e6:.6g} MHz')


class CavTransmission:
    """Fit a two-port cavity transmission PEAK using complex S21.

    Frequencies must be in Hz. Fits QL, f0, magBack, phaseOff, eDelay.
    QL^-1 = Qint^-1 + Qext1^-1 + Qext2^-1; only QL is identifiable
    when the complex peak amplitude is free. magBack is the fitted peak
    amplitude, not an off-resonance background.
    The multiplicative factor is
    magBack*exp(1j*(phaseOff + 2*pi*eDelay*(f-f0))).
    eDelay is a signed phase slope in seconds. Set fit_delay=True to fit it.
    The model assumes negligible additive/direct-path transmission.
    """

    def __init__(self, coordinates, data, conjugate=True, fit_delay=False):
        freq = np.asarray(coordinates, dtype=float)
        values = np.asarray(data, dtype=complex)
        if freq.ndim != 1 or values.shape != freq.shape or len(freq) < 5:
            raise ValueError('Frequency and complex S21 must be matching 1D arrays with >=5 points.')
        if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(values)):
            raise ValueError('Frequency and S21 must be finite.')
        order = np.argsort(freq)
        self.coordinates, self.data = freq[order], values[order]
        if np.any(np.diff(self.coordinates) <= 0) or np.any(self.coordinates <= 0):
            raise ValueError('Frequency samples must be unique and positive, in Hz.')
        if not np.any(np.abs(values) > 0):
            raise ValueError('S21 is identically zero.')
        self.conjugate = conjugate
        self.fit_delay = fit_delay

    def model(self, coordinates, QL, f0, magBack, phaseOff, eDelay=0):
        response = cav_trans_func(coordinates, QL, f0)
        if self.conjugate:
            response = response.conjugate()
        return response * magBack * np.exp(1j * (
            phaseOff + TWOPI * eDelay * (coordinates - f0)))

    def guess(self, coordinates, data):
        freq, data = np.asarray(coordinates), np.asarray(data)
        amp = np.abs(data)
        phase = np.unwrap(np.angle(data))
        idx = np.argmax(amp)
        f0 = freq[idx]
        amplitude, offset = amp[idx], phase[idx]
        target = amplitude / np.sqrt(2)
        inside = amp >= target
        delay = 0.0
        left = right = int(idx)
        while left > 0 and inside[left-1]:
            left -= 1
        while right < len(freq)-1 and inside[right+1]:
            right += 1
        width = freq[right]-freq[left]
        if width <= 0 or left == 0 or right == len(freq)-1:
            width = np.sqrt((freq[-1]-freq[0]) * np.min(np.diff(freq)))
        QL = f0 / width
        if self.fit_delay:
            resonance = cav_trans_func(freq, QL, f0)
            if self.conjugate:
                resonance = resonance.conjugate()
            # Use the central, higher-SNR part of the peak for initialization.
            good = amp > amplitude * 0.2
            if np.count_nonzero(good) >= 5:
                residual_phase = np.unwrap(np.angle(data[good] / resonance[good]))
                slope, offset = np.polyfit(freq[good]-f0, residual_phase, 1)
                delay = slope / TWOPI
        parameters = lmfit.Parameters()
        parameters.add('f0', value=f0, min=freq[0], max=freq[-1])
        parameters.add('magBack', value=amplitude, min=np.finfo(float).eps)
        parameters.add('phaseOff', value=offset)
        parameters.add('eDelay', value=delay, vary=self.fit_delay)
        parameters.add('QL', value=QL, min=1)
        return parameters

    def run(self, *args, params=None, dry=False, **kwargs):
        """Fit with lmfit; params overrides may be Parameters, dict or numbers.

        dry=True returns the guessed/overridden model without optimization.
        Other keywords, e.g. method or weights, are passed to Model.fit.
        """
        parameters = self.guess(self.coordinates, self.data)
        if params is not None:
            for name, value in params.items():
                if name not in parameters:
                    raise ValueError(f'Unknown transmission parameter: {name}')
                if isinstance(value, lmfit.Parameter):
                    parameters[name] = value
                elif isinstance(value, dict):
                    parameters[name].set(**value)
                else:
                    parameters[name].set(value=value)
        model = lmfit.Model(self.model, independent_vars=['coordinates'])
        if dry:
            result = ModelResult(model, parameters, data=self.data,
                                 fcn_kws={'coordinates': self.coordinates})
            result.best_fit = model.eval(params=parameters, coordinates=self.coordinates)
        else:
            result = model.fit(self.data, parameters, *args,
                               coordinates=self.coordinates, **kwargs)
        return CavTransmissionResult(result)




if __name__ == '__main__':
    import h5py
    filepath = r'L:\Data\WISPE3D\Modes\20210809\CavModes\Cav'
    (freq, real, imag, mag, phase) = getVNAData(filepath, plot=0)

    cavRef = CavReflection(freq, real + 1j * imag)
    results = cavRef.run(params={"Qext": lmfit.Parameter("Qext", value=3.48354e+03), "Qint": lmfit.Parameter("Qint", value=7e3)})
    results.plot()
    results.print()
    print(cavRef.guess(cavRef.coordinates, cavRef.data))

    # results = cavRef.run(dry=True, params={"Qext": lmfit.Parameter("Qext", value=3.48354e+03), "Qint": lmfit.Parameter("Qint", value=7e3)})
    # results.lmfit_result.plot()

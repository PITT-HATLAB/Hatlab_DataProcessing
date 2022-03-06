from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import lmfit

from Hatlab_DataProcessing.fitter.fitter_base import Fit, FitResult


class Linear(Fit):
    @staticmethod
    def model(coordinates, k, b) -> np.ndarray:
        """$ k * x + b $"""
        return k * coordinates + b

    @staticmethod
    def guess(coordinates, data):
        k = (data[-1] - data[0]) / (coordinates[-1] - coordinates[0])
        b = data[-1] - k * data[0]
        return dict(k=k, b=b)


class Cosine(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, of) -> np.ndarray:
        """$A cos(2 pi f x + phi) + of$"""
        return A * np.cos(2 * np.pi * coordinates * f + phi) + of

    @staticmethod
    def guess(coordinates, data):
        of = np.mean(data)
        A = (np.max(data) - np.min(data)) / 2.

        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])

        return dict(A=A, f=f, phi=phi, of=of)


class ExponentialDecay(Fit):
    @staticmethod
    def model(coordinates, A, tau, of) -> np.ndarray:
        """ A * exp (-x/tau) + of"""
        return A * np.exp(-coordinates / tau) + of

    @staticmethod
    def guess(coordinates, data):
        of = data[-1]
        A = data[0] - data[-1]
        tau_ = (coordinates[-1] - coordinates[0])/3
        tau = lmfit.Parameter("tau", value=tau_,  min=0.0000001)
        return dict(A=A, tau=tau, of=of)


class ExponentialDecayWithCosine(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, tau, of) -> np.ndarray:
        """ A * cos(2 pi f x + phi) * exp (-x/tau) + of"""
        return A * np.cos(f * np.pi * 2 * coordinates + phi) * np.exp(-coordinates / tau) + of

    @staticmethod
    def guess(coordinates, data):
        of = np.mean(data)
        A = (np.max(data) - np.min(data)) / 2.
        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])
        tau_ = (1 / 4.0) * (coordinates[-1] - coordinates[0])
        tau = lmfit.Parameter("tau", value=tau_, min=0.0001)
        return dict(A=A, f=f, phi=phi, tau=tau, of=of)


class ExponentialDecayWithCosineSquare(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, tau, of) -> np.ndarray:
        """ A * cos(2 pi f x + phi) ** 2 * exp (-x/tau) + of"""
        return A * np.cos(f * np.pi * 2 * coordinates + phi) ** 2 * np.exp(-coordinates / tau) + of

    @staticmethod
    def guess(coordinates, data):
        of = np.min(data)
        A = (np.max(data) - np.min(data))
        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size, np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx] / 2
        phi = np.angle(fft_val[idx]) / 2
        tau_ = (1 / 4.0) * (coordinates[-1] - coordinates[0])
        tau = lmfit.Parameter("tau", value=tau_, min=0.0001)
        return dict(A=A, f=f, phi=phi, tau=tau, of=of)


if __name__ == "__main__":
    fileName = "Q1"
    filePath = r"C:/Users/hatla/Downloads//"
    from Hatlab_DataProcessing.analyzer.rotateIQ import RotateData
    import json

    with open(filePath + fileName+"_piPulse", 'r') as infile:
        dataDict = json.load(infile)
    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])

    rotIQ = RotateData(x_data, i_data + 1j * q_data)
    iq_new = rotIQ.run()
    iq_new.plot()

    # fit cosine
    cosFit = Cosine(x_data, iq_new.params["i_data"].value)
    # cosFitResult = cosFit.run(params={"A":lmfit.Parameter("A", 1, vary=False)}) # example of adjusting fitting parameter
    cosFitResult = cosFit.run()
    cosFitResult.plot()


    # fit decay
    with open(filePath + fileName+"_t1", 'r') as infile:
        dataDict = json.load(infile)
    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])

    rotIQ = RotateData(x_data, i_data + 1j * q_data)
    iq_new = rotIQ.run(angle=iq_new.params["rot_angle"].value)
    iq_new.plot()

    decayFit = ExponentialDecay(x_data, iq_new.params["i_data"].value)
    # cosFitResult = cosFit.run(params={"A":lmfit.Parameter("A", 1, vary=False)}) # example of adjusting fitting parameter
    decayFitResult = decayFit.run()
    decayFitResult.plot()


    # fit ramsey
    with open(filePath + fileName+"_t2R", 'r') as infile:
        dataDict = json.load(infile)
    x_data = dataDict["x_data"]
    i_data = np.array(dataDict["i_data"])
    q_data = np.array(dataDict["q_data"])

    rotIQ = RotateData(x_data, i_data + 1j * q_data)
    iq_new = rotIQ.run(angle=iq_new.params["rot_angle"].value)
    iq_new.plot()

    ramseyFit = ExponentialDecayWithCosine(x_data, iq_new.params["i_data"].value)
    # cosFitResult = cosFit.run(params={"A":lmfit.Parameter("A", 1, vary=False)}) # example of adjusting fitting parameter
    ramseyFitResult = ramseyFit.run()
    ramseyFitResult.plot()
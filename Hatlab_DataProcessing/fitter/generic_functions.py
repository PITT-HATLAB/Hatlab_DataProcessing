from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import lmfit

from fitter_base import Fit, FitResult


class Cosine(Fit):
    @staticmethod
    def model(coordinates, A, f, phi, of) -> np.ndarray:
        """$A \cos(2 \pi f x + \phi) + of$"""
        return A * np.cos(2 * np.pi * coordinates * f + phi) + of

    @staticmethod
    def guess(coordinates, data):
        of = np.mean(data)
        A = (np.max(data) - np.min(data)) / 2.

        fft_val = np.fft.rfft(data)[1:]
        fft_frq = np.fft.rfftfreq(data.size,
                                  np.mean(coordinates[1:] - coordinates[:-1]))[1:]
        idx = np.argmax(np.abs(fft_val))
        f = fft_frq[idx]
        phi = np.angle(fft_val[idx])

        return dict(A=A, f=f, phi=phi, of=of)

class Exponential(Fit):
    @staticmethod
    def model(coordinates, a, b) -> np.ndarray:
        """ a * b ** x"""
        return a * b ** coordinates

    @staticmethod
    def guess(coordinates, data):
        return dict(a=1, b=2)


if __name__=="__main__":
    fileName = "Q1_piPulse"
    filePath = r"C:/Users/hatla/Downloads//"
    from Hatlab_DataProcessing.analyzer.rotateIQ import RotateData
    import json
    with open(filePath + fileName, 'r') as infile:
        dataDict = json.load(infile)

    x_data = dataDict["x_data"]
    i_data = dataDict["i_data"]
    q_data = dataDict["q_data"]

    rotIQ = RotateData(i_data, q_data)
    iq_new = rotIQ.run()
    iq_new.plot(x_data)

    cosFit = Cosine(x_data, iq_new.params["i_data"].value)
    # cosFitResult = cosFit.run(params={"A":lmfit.Parameter("A", 1, vary=False)}) # example of adjusting fitting parameter
    cosFitResult = cosFit.run() # example of adjusting parameter
    cosFitResult.plot()
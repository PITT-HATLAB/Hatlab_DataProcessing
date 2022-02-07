from typing import Tuple, Any, Optional, Union, Dict, List, Literal

import numpy as np
from matplotlib import pyplot as plt
import json

from Hatlab_DataProcessing.base import Analysis, AnalysisResult

class RotateResult(AnalysisResult):
    def plot(self,x_data=None, figName=None):
        if x_data is None:
            x_data = np.arange(len(self.params['i_data'].value))
        plt.figure(figName)
        plt.plot(x_data, self.params['i_data'].value)
        plt.plot(x_data, self.params['q_data'].value)


class RotateData(Analysis):
    """
    rotate the iq data in rad units, return result class that contains new IQ data and rotation angle.
    """
    @staticmethod
    def analyze(i_data, q_data, angle:Union[float,Literal["find"]]="find"):
        i_data = np.array(i_data)
        q_data = np.array(q_data)
        deriv = []
        if angle == "find":
            for i in range(2001):
                angle = 0.001 * i
                i_temp, q_temp = rotate_complex(i_data, q_data, angle)
                line_fit = np.zeros(len(q_temp)) + q_temp.mean()
                deriv_temp = ((q_temp - line_fit) ** 2).sum()
                deriv.append(deriv_temp)
                final = 0.001 * np.argwhere(np.array(deriv) == np.min(np.array(deriv)))
                rotation_angle = final.ravel()[0]
        elif type(angle) in [float, np.float]:
            rotation_angle = angle
        i_new, q_new = rotate_complex(i_data, q_data, rotation_angle)
        print(rotation_angle)
        return RotateResult(dict(i_data=i_new,q_data=q_new, rot_angle=rotation_angle))


def rotate_complex(real_part, imag_part, angle):
    """
    rotate the complex number as rad units.
    """
    iq_new = (real_part + 1j * imag_part) * np.exp(1j * np.pi * angle)
    return iq_new.real, iq_new.imag


if __name__=="__main__":
    fileName = "Q1_piPulse"
    filePath = r"C:/Users/hatla/Downloads//"
    with open(filePath + fileName, 'r') as infile:
        dataDict = json.load(infile)

    x_data = dataDict["x_data"]
    i_data = dataDict["i_data"]
    q_data = dataDict["q_data"]

    rotIQ = RotateData(i_data, q_data)
    iq_new = rotIQ.run()
    iq_new.plot(x_data)

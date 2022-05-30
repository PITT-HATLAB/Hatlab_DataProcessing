from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.model import ModelResult
from Hatlab_DataProcessing.fitter.fitter_base import Fit, FitResult
from Hatlab_DataProcessing.helpers.unit_converter import freqUnit, rounder, realImag2magPhase
from scipy.ndimage import gaussian_filter

TWOPI = 2 * np.pi
PI = np.pi


def twoD_gaussian_func(coord: tuple, amp, x0, y0, sigmaX, sigmaY, theta, offset):
    """ 2D gaussian function
        https://en.wikipedia.org/wiki/Gaussian_function
    """
    (x, y) = coord
    a = (np.cos(theta) ** 2) / (2 * sigmaX ** 2) + (np.sin(theta) ** 2) / (2 * sigmaY ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigmaX ** 2) + (np.sin(2 * theta)) / (4 * sigmaY ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigmaX ** 2) + (np.cos(theta) ** 2) / (2 * sigmaY ** 2)
    z = offset + amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                                 + c * ((y - y0) ** 2)))
    return z


def guess_gau2D_params(x, y, z, nBlobs, maskIndex=None):
    """ find the centers of gaussian blobs by looking for peaks of the 2d histogram,
        assume each blob is seperated by maskIndex

    :param x: bin edges along the first dimention
    :param y: bin edges along the second dimention
    :param z: The bi-dimensional histogram of samples x and y
    :param nBlobs: number of gaussian blobs
    :param maskIndex: distance between each blob
    :return:
    """
    border = np.max(np.abs(np.array([x, y])))
    if maskIndex is None:
        maskIndex = int(border//200)

    ampList = np.zeros(nBlobs)
    x0List = np.zeros(nBlobs)
    y0List = np.zeros(nBlobs)
    sigmaXList = np.zeros(nBlobs) + border / 4
    sigmaYList = np.zeros(nBlobs) + border / 4
    thetaList = np.zeros(nBlobs)
    offsetLost = np.zeros(nBlobs)

    for i in range(nBlobs):
        x1indx, y1indx = np.unravel_index(np.argmax(z, axis=None), z.shape)
        x1ini, y1ini = x[x1indx, y1indx], y[x1indx, y1indx]
        amp1 = np.max(z)
        mask1 = np.zeros((len(x) , len(y) ))
        mask1[-maskIndex + x1indx:maskIndex + x1indx, -maskIndex + y1indx:maskIndex + y1indx] = 1
        z = np.ma.masked_array(z, mask=mask1)

        ampList[i] = amp1
        x0List[i], y0List[i] = x1ini, y1ini

    return ampList, x0List, y0List, sigmaXList, sigmaYList, thetaList, offsetLost


class Gaussian2DResult(FitResult):
    def __init__(self, lmfit_result: lmfit.model.ModelResult, coord, data):
        self.lmfit_result = lmfit_result
        self.coord = coord
        self.data = data
        self.params = lmfit_result.params
        for p in self.params:
            self.__setattr__(p, self.params[p].value)
        self.sigma_g = np.sqrt(self.sigmaX1 ** 2 + self.sigmaY1 ** 2)
        self.sigma_e = np.sqrt(self.sigmaX2 ** 2 + self.sigmaY2 ** 2)
        self.ImOverSigma = np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) / self.sigma_g

    def plot(self, **figArgs):
        x, y = self.coord
        z = gaussian_filter(self.data, [2, 2])
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(x, y, z)
        ax.set_aspect(1)
        ax.contour(x, y, self.lmfit_result.best_fit,  colors='w')

    def print(self):
        for p in self.params:
            print(f"{p}: {np.round(self.params[p].value, 4)}")



class Gaussian2D_Base(Fit):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, nBlobs=2, maskIndex=None):
        """ fit multiple 2D gaussian blobs
        :param nBlobs:  number of gussian blobs
        """
        self.coordinates = coordinates
        self.data = data
        self.nBlobs = nBlobs
        self.maskIndex = maskIndex
        self.pre_process()

    def guess(self, coordinates, data):
        border = np.max(np.abs(coordinates))
        if self.maskIndex is None:
            self.maskIndex = int(border // 200)

        (x, y) = coordinates
        z = gaussian_filter(data, [2, 2])

        ampList, x0List, y0List, sigmaXList, sigmaYList, thetaList, offsetLost = \
            guess_gau2D_params(x, y, z, self.nBlobs, self.maskIndex)

        params = lmfit.Model(self.model).make_params()
        paramDict = dict(params)

        for i in range(self.nBlobs):
            paramDict[f"amp{i + 1}"] = lmfit.Parameter(f"amp{i + 1}", value=ampList[i], min=0,
                                                       max=ampList[0] * 2)
            paramDict[f"x{i + 1}"] = lmfit.Parameter(f"x{i + 1}", value=x0List[i], min=-border,
                                                     max=border)
            paramDict[f"y{i + 1}"] = lmfit.Parameter(f"y{i + 1}", value=y0List[i], min=-border,
                                                     max=border)
            paramDict[f"sigmaX{i + 1}"] = lmfit.Parameter(f"sigmaX{i + 1}", value=sigmaXList[i],
                                                          min=0, max=border / 2)
            paramDict[f"sigmaY{i + 1}"] = lmfit.Parameter(f"sigmaY{i + 1}", value=sigmaYList[i],
                                                          min=0, max=border / 2)
            paramDict[f"theta{i + 1}"] = lmfit.Parameter(f"theta{i + 1}", value=thetaList[i], min=0,
                                                         max=TWOPI)
            paramDict[f"offset{i + 1}"] = lmfit.Parameter(f"offset{i + 1}", value=offsetLost[i],
                                                          min=0, max=ampList[0] / 5)

        return paramDict

    def run(self, *args: Any, **kwargs: Any) -> Gaussian2DResult:
        lmfit_result = self.analyze(self.coordinates, self.data, *args, **kwargs)
        return Gaussian2DResult(lmfit_result, self.coordinates, self.data)

        #todo : reorder the fit result in order of amp
        gIndex = np.argmax(np.array([amp1, amp2, amp3]))
        eIndex = np.argmax(
            np.ma.masked_values(np.array([amp1, amp2, amp3]), np.array([amp1, amp2, amp3])[gIndex]))
        fIndex = np.argmin(np.array([amp1, amp2, amp3]))
        gef_order = [gIndex, eIndex, fIndex]
        # if y1 < y2:
        #     [x1, y1, amp1, sigma1x, sigma1y, x2, y2, amp2, sigma2x, sigma2y] = [x2, y2, amp2, sigma2x, sigma2y, x1, y1, amp1, sigma1x, sigma1y]
        gef_xy = np.array([[x1, y1], [x2, y2], [x3, y3]])[gef_order]
        gef_sigma = np.array([[sigma1x, sigma1y], [sigma2x, sigma2y], [sigma3x, sigma3y]])[
            gef_order]
        gef_amp = np.array([amp1, amp2, amp3])[gef_order]


        # x, y = self.coord
        # z = gaussian_filter(self.data, [2, 2])
        # fig, ax = plt.subplots(1, 1)
        # ax.pcolormesh(x, y, z)
        # ax.set_aspect(1)
        # ax.contour(x, y, self.lmfit_result.best_fit,  colors='w')
        # ax.scatter(*gef_xy.transpose(), c="r", s=0.7)
        # for i, txt in enumerate(["g", "e", "f"]):
        #     ax.annotate(txt, (gef_xy[i][0], gef_xy[i][1]))


class Gaussian2D_2Blob(Gaussian2D_Base):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, maskIndex=None):
        super().__init__(coordinates, data, 2, maskIndex)

    @staticmethod
    def model(coordinates,
              amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1,
              amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2):
        """"multiple 2D gaussian function"""
        z = twoD_gaussian_func(coordinates, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1) + \
            twoD_gaussian_func(coordinates, amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2)
        return z


class Gaussian2D_3Blob(Gaussian2D_Base):
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray],
                 data: np.ndarray, maskIndex=None):
        super().__init__(coordinates, data, 3, maskIndex)

    @staticmethod
    def model(coordinates,
              amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1,
              amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2,
              amp3, x3, y3, sigmaX3, sigmaY3, theta3, offset3):
        """"multiple 2D gaussian function"""
        z = twoD_gaussian_func(coordinates, amp1, x1, y1, sigmaX1, sigmaY1, theta1, offset1) + \
            twoD_gaussian_func(coordinates, amp2, x2, y2, sigmaX2, sigmaY2, theta2, offset2) + \
            twoD_gaussian_func(coordinates, amp3, x3, y3, sigmaX3, sigmaY3, theta3, offset3)
        return z

if __name__ == '__main__':
    pass

import numpy as np
import matplotlib.pyplot as plt
from Hatlab_DataProcessing.fitter.arb_gaussian import classify_point, peakfinder_2d

'''
Some functions for extracting basic information from a single set of IQ traces that correspond to resonator states.
For example
- state location on IQ plane
- squeezing degree and angle of coherent state
'''

def blob_info(idata, qdata, plot=False):

    plt.figure()
    zz, x, y = np.histogram2d(idata, qdata, bins=np.sqrt(len(idata)))


    x = (x[0:-1] + x[1:]) / 2
    y = (y[0:-1] + y[1:]) / 2
    xx, yy = np.meshgrid(x, y)


    idxx, idxy, heights = peakfinder_2d(zz, radius=6, threshold=10)

    if plot:
        plt.pcolor(x, y, np.log(zz))
        plt.colorbar()
        plt.scatter(x[idxx], y[idxy], color='r')
        heights = zz[idxy, idxx]

    return idxx, idxy, heights



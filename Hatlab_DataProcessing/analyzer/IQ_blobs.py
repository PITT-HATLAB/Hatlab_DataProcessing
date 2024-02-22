import numpy as np
import matplotlib.pyplot as plt
from Hatlab_DataProcessing.fitter.arb_gaussian import classify_point, peakfinder_2d, fit_arb_gaussians

'''
Some functions for extracting basic information from a single set of IQ traces that correspond to resonator states.
For example
- state location on IQ plane
- squeezing degree and angle of coherent state
'''

def blob_info(idata, qdata, num_states, HEMT_sigma, plot=False):

    range = np.sort(np.abs(idata + 1j*qdata))[-len(idata)//10]*1.2
    print(range)

    # range = np.max([np.max(np.abs(idata)),np.max(np.abs(qdata))])

    # bins = np.min([len(np.arange(-range,range,HEMT_sigma/5)),101])
    bins = int(np.sqrt(len(idata)))

    bins = 101

    print(bins)

    zz, x, y = np.histogram2d(idata.flatten(), qdata.flatten(), bins=bins, range=[[-range,range],[-range,range]])

    zz = zz.copy()

    x = (x[0:-1] + x[1:]) / 2
    y = (y[0:-1] + y[1:]) / 2
    xx, yy = np.meshgrid(x, y)


    dx = x[1]-x[0]

    radius = np.max([5,np.min([int(HEMT_sigma/dx),bins//2])])  # not too big, not too small

    idxx, idxy, heights = peakfinder_2d(zz, radius, num_states, plot=False)

    sigma_guess = np.max([HEMT_sigma, dx*5])

    fitted_params, ax = fit_arb_gaussians(x, y, zz.transpose(), idxx, idxy, heights, sigma_guess, plot=plot)

    if plot:
        ax.scatter(x[idxx], y[idxy], color='r', alpha=0.5)

    return fitted_params



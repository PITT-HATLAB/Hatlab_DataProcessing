import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

'''
More general gaussian fitting and classification schemes.

You can think of three independent steps implemented here.

1) Peak finding: given some histogram data, where are the peaks?
2) Peak fitting: given the peaks, what is the peak width sigma (and how skewed is it, and along what axis)?
3) Classification: given previous information, how to assign a new point on the histogram to a peak 
'''

def peakfinder_2d(zz, radius, num_peaks):
    '''
    The fastest way I can think of without a just-in-time compiler. You can imagine that each point checks for
    neighboring points (radius r) and calls itself a peak if it's bigger than all its neighbors, not including
    edges. It's done with array slicing rather than explicit loop.

    Faster than looping to each point in the 2d array and comparing, but not way faster.

    :param zz: 2d data
    :param radius: Distance to check for higher neighbors
    :param num_peaks: Only take the largest of the detected peaks (largest value).
    '''

    neighbors = []

    for i in range(0, radius * 2):
        for j in range(0, radius * 2):

            if (i != radius or j != radius):
                neighbor = zz[i:-radius * 2 + i,
                           j:-radius * 2 + j]  # not necessarily nearest neighbor if radius > 1

                neighbors.append(neighbor)

        neighbor = zz[i:-radius * 2 + i, radius * 2:]

        neighbors.append(neighbor)

    for j in range(0, radius * 2):
        neighbor = zz[radius * 2:, j:-radius * 2 + j]

        neighbors.append(neighbor)

    neighbor = zz[radius * 2:, radius * 2:]

    neighbors.append(neighbor)

    neighbors = np.array(neighbors)

    max_neighbors = zz * 0 + np.max(zz)
    max_neighbors[radius:-radius, radius:-radius] = np.max(neighbors, axis=0)

    idx = np.where(max_neighbors < zz)  # identifies the peaks (i.e., finds their indices)

    idxx = idx[0]
    idxy = idx[1]

    heights = zz[idxx, idxy]

    order = np.flip(np.argsort(heights))

    idxx = idxx[order]
    idxy = idxy[order]

    # only takes the tallest peaks, according to the requested number of states

    if num_peaks != None:
        idxx = idxx[0:num_peaks]
        idxy = idxy[0:num_peaks]
        heights = heights[0:num_peaks]


    if True:
        plt.figure()
        plt.pcolor(zz)
        idxx, idxy, max_neighbors = peakfinder_2d(zz, radius=6, threshold=10)
        plt.scatter(idxx, idxy, color='r')

    return idxx, idxy, heights, max_neighbors



def fit_arb_gaussians(x, y, zz, idxx, idxy, heights):
    NUM_PARAMS = 6

    def gaussians_2d(xx, yy, As, x0s, y0s, rs, skews, thetas):

        '''
        model function for a set of gaussians
        '''

        zz = xx * 0

        for i in range(0, len(As)):
            zz += gaussian_2d(xx, yy, As[i], x0s[i], y0s[i], rs[i], skews[i], thetas[i])

        return zz

    def gaussian_2d(xx, yy, A, x0, y0, r, skew, theta):
        '''
        model function for a single 2D Gaussian function
        '''

        xx_n = (xx - x0) * np.cos(theta) + (yy - y0) * np.sin(theta)
        yy_n = (yy - y0) * np.cos(theta) - (xx - x0) * np.sin(theta)

        return A * np.exp(- xx_n ** 2 / (r * (skew + 1)) ** 2 - yy_n ** 2 / (r / (skew + 1)) ** 2)

    def fit_cost(params, xx, yy, data):
        '''
        the cost function to minimize (least squares error)
        '''

        params_s = np.reshape(params, (NUM_PARAMS, len(params) // NUM_PARAMS))
        As = params_s[0, :]
        x0s = params_s[1, :]
        y0s = params_s[2, :]
        rs = params_s[3, :]
        skews = params_s[4, :]
        thetas = params_s[5, :]
        model = gaussians_2d(xx, yy, As, x0s, y0s, rs, skews, thetas)

        return np.sum((model - data) ** 2)

    def make_initial_guess(x, y, idxx, idxy, heights):

        ones = np.ones(len(idxx))

        if len(idxx) == 1:
            initial_guess = np.array([heights, x[idxx], y[idxy], ones, ones / 10, ones / 10])

        else:

            avg_dist = np.repeat(np.mean(np.sqrt(np.diff(x[idxx]) ** 2 + np.diff(y[idxy]) ** 2), axis=0), len(idxx))

            initial_guess = np.array([heights, x[idxx], y[idxy], avg_dist / 10, ones / 10, ones / 10])

        return initial_guess

    def make_bounds(initial_guess):
        initial_guess_flat = initial_guess.flatten()
        print(initial_guess_flat)
        bounds = []

        for i in range(0, len(initial_guess_flat)):

            if i % 6 == 0:  # peak height
                bounds.append([initial_guess_flat[i] * 0.8, initial_guess_flat[i] * 1.2])

            if i % 6 == 1:  # x position
                bounds.append([initial_guess_flat[i] * 0.9, initial_guess_flat[
                    i] * 1.1])  # we already know the position pretty well, so narrow bounds

            if i % 6 == 2:  # y position
                bounds.append([initial_guess_flat[i] * 0.9, initial_guess_flat[
                    i] * 1.1])  # we already know the position pretty well, so narrow bounds

            if i % 6 == 3:  # radius
                bounds.append([0, 10000])

            if i % 6 == 4:  # skew
                bounds.append([0, 10])

            if i % 6 == 5:  # theta
                bounds.append([None, None])

        print(bounds)
        return bounds

    initial_guess = make_initial_guess(x, y, idxx, idxy, heights)

    bounds = make_bounds(initial_guess)

    # Perform the minimization
    result = minimize(fit_cost, initial_guess.flatten(), args=(xx, yy, zz), method='Nelder-Mead')

    # Extract the fitted parameters
    fitted_params = np.reshape(result.x, (NUM_PARAMS, len(result.x) // NUM_PARAMS))

    # Define a function to plot the data and the fitted Gaussian
    def plot_data_and_fit(data, fitted_params):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        # Plot the original data
        ax.pcolor(x, y, data, cmap='viridis')
        ax.set_title('Original Data')

        # Create a grid of (x, y) points for the fitted Gaussian model
        x_fit, y_fit = np.meshgrid(x, y)
        for i in range(0, len(idxx)):
            fitted_data = gaussian_2d(x_fit, y_fit, *fitted_params[:, i])
            initial_data = gaussian_2d(x_fit, y_fit, *initial_guess[:, i])

            # ax[1].pcolor(x,y,fitted_data, cmap='viridis',clim=[0,3])
            # ax[1].set_title('Fitted Data')

            # Plot the fitted Gaussian model
            CS = ax.contour(x, y, fitted_data, origin='lower', colors='w', levels=3, linestyles='-')
            # CS = ax.contour(x,y,initial_data, origin='lower', colors='k', levels=3,linestyles='-')
            ax.set_title('Fitted Gaussian Model')
            ax.clabel(CS, inline=True, fontsize=10)

        plt.show()

    # Plot the original data and the fitted Gaussian model
    # plot_data_and_fit(zz, initial_guess)

    plot_data_and_fit(zz, fitted_params)

    print("Fitted Parameters:")
    print("A:", fitted_params[0])
    print("x0:", fitted_params[1])
    print("y0:", fitted_params[2])
    print("r:", fitted_params[3])
    print("skew:", fitted_params[4])
    print("theta:", fitted_params[5])

    return fitted_params

def classify_point(x_points, y_points, x_peaks, y_peaks, heights=None, plot=False):
    '''
    For a given list of points, this tells you the nearest state, and so returns a list
    of labels, corresponding to the order that the peaks are provided in.

    The classification is weighted if heights are supplied
    '''

    if heights is None:
        weights = x_peaks * 0 + 1
    else:
        weights = 1 / heights.copy()

    distances = np.zeros([len(x_points), len(x_peaks)])

    for i in range(0, len(x_peaks)):
        distances[:, i] = np.sqrt((x_points - x_peaks[i]) ** 2 + (y_points - y_peaks[i]) ** 2) * weights[i]

    states = np.argsort(distances, axis=1)[:, 0]

    if plot:

        range_lim = np.max(np.max(np.abs(x_peaks)),np.max(np.abs(y_peaks)))*1.2

        range = [[-range_lim,range_lim],[-range_lim,range_lim]]

    return states


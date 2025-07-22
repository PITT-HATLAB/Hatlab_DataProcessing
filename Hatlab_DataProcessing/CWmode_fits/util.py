import numpy as np

def overlapping_support_points(array1, array2):
    '''
    interesting little interpolation problem for the g/Delta fit

    the sweep data might be taken on different grids. To get around
    this, I take the overlapping range of the sweep values, and take
    the union of the sets of points in that overlapping
    range. Those are the points at which the interpolation is evaluated.
    '''

    array_min = np.max([np.min(array1), np.min(array2)])
    array_max = np.min([np.max(array1), np.max(array2)])

    array1_trimmmed = array1[np.where((array1 >= array_min) * (array1 <= array_max))] # use Boolean multiplication as logical AND
    array2_trimmmed = array2[np.where((array2 >= array_min) * (array2 <= array_max))]

    array_out = np.concatenate([array1_trimmmed, array2_trimmmed])

    array_out = np.sort(np.unique(array_out))

    return array_out
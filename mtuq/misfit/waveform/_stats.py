
import numpy as np

from copy import deepcopy
from mtuq.util import warn
from mtuq.util.math import list_intersect_with_indices
from mtuq.util.signal import get_components


def calculate_norm_data(data, norm, components):
    # error checking
    assert norm in ('L1', 'L2')

    norm_data = 0.
    for _j, d in enumerate(data):
        _components, indices = list_intersect_with_indices(
            get_components(d), components)

        if not indices:
            continue

        # time sampling scheme
        npts = d[0].data.size
        dt = d[0].stats.delta

        for _k in indices:
            r = d[_k].data

            if norm=='L1':
                norm_data += np.sum(np.abs(r))*dt

            elif norm=='L2':
                norm_data += np.sum(r**2)*dt

    return norm_data


def estimate_sigma(data, greens, best_source, norm, components,
    time_shift_min, time_shift_max):

    # error checking
    assert norm in ('L1', 'L2')

    residuals = []
    for _j, d in enumerate(data):
        _components, indices = list_intersect_with_indices(
            get_components(d), components)

        if not indices:
            continue

        # generate synthetics
        greens[_j]._set_components(get_components(d))
        s = greens[_j].get_synthetics(best_source)

        # time sampling scheme
        npts = d[0].data.size
        dt = d[0].stats.delta

        padding_left = int(+time_shift_max/dt)
        padding_right = int(-time_shift_min/dt)
        npts_padding = padding_left + padding_right

        # array to hold cross correlations
        corr = np.zeros(npts_padding+1)

        #
        # calculate residuals
        # 

        corr[:] = 0.
        for _k in indices:
            corr += np.correlate(s[_k].data, d[_k].data, 'valid')

        npts_shift = padding_left - corr.argmax()
        time_shift = npts_shift*dt

        # what start and stop indices will correctly shift synthetics
        # relative to data?
        start = padding_left - npts_shift
        stop = start + npts

        for _k in indices:

            # substract data from shifted synthetics
            r = s[_k].data[start:stop] - d[_k].data

            # sum the resulting residuals
            if norm=='L1':
                residuals += [np.sum(np.abs(r))*dt]

            elif norm=='L2':
                residuals += [np.sum(r**2)*dt]


    return np.mean(residuals)**0.5


"""
Data misfit module (fast pure Python version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
from mtuq.util import iterable
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import autocorr_nd1, autocorr_nd2, corr_nd1_nd2, corr_init
from mtuq.util.signal import get_components, get_time_sampling
from scipy.signal import fftconvolve



def misfit(data, greens, sources, norm, time_shift_groups, time_shift_max, 
    verbose=0):
    """
    Data misfit function (fast pure Python version)

    See ``mtuq/misfit/__init__.py`` for more information
    """
    sources = iterable(sources)
    results = np.zeros((len(sources), 1))

    #
    # initialize Green's function machinery
    #
    for _j, d in enumerate(data):
        greens[_j].reset_components(get_components(d))

    #
    # cross-correlations are a major part of the numerical work
    #
    corr_sum = corr_init(data, time_shift_max)
    data_greens = corr_nd1_nd2(data, greens, time_shift_max)

    if norm in ['L2', 'hybrid']:
        data_data = autocorr_nd1(data, time_shift_max)
        greens_greens = autocorr_nd2(greens, time_shift_max)

    #
    # begin iterating over sources
    #
    for _i, source in enumerate(sources):
        for _j, d in enumerate(data):

            components = greens[_j].components
            if not components:
                continue

            if norm=='L1':
                s = greens[_j].get_synthetics(source)

            # time sampling scheme
            npts, dt = get_time_sampling(d)
            npts_padding = int(time_shift_max/dt)


            #
            # evaluate misfit for all components at given station
            # 
            for group in time_shift_groups:
                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all comonents in a
                # given group, subject to time_shift_max constraint
                _, indices = list_intersect_with_indices(
                    components, group)

                npts_shift = get_time_shift(
                    data_greens[_j], corr_sum[_j], source, indices)

                time_shift = npts_shift*dt

                # what start and stop indices will correctly shift synthetics
                # relative to data?
                start = npts_padding - npts_shift
                stop = npts+npts_padding - npts_shift

                for _k in indices:
                    misfit = 0.

                    # sum the resulting residuals
                    if norm=='L1':
                        r = s[_k].data[start:stop] - d[_k].data
                        misfit = dt*np.sum(abs(r))

                    elif norm=='L2':
                        misfit = dt * get_L2_norm(
                            greens_greens[_j], data_data[_j], data_greens[_j],
                            source, _k, npts_shift+npts_padding)

                    elif norm=='hybrid':
                        misfit = dt * get_L2_norm(
                            greens_greens[_j], data_data[_j], data_greens[_j],
                            source, _k, npts_shift+npts_padding)**0.5

                    results[_i] += d[_k].weight * misfit

    return results


def get_L2_norm(greens_greens, data_data, data_greens, source, i1, i2):
    """
    Calculates L2 norm of data and shifted synthetics using
    ||s - d||^2 = s^2 + d^2 - 2sd
    """
    misfit = 0

    # calculate d^2 
    misfit += data_data[i1]

    # calculate s^2 
    misfit += np.dot(np.dot(greens_greens[i1, i2, :, :], source), source)

    # calculate sd 
    misfit -= 2*np.dot(data_greens[i1, :, i2], source)

    return misfit


def debug_L2_norm(data, synthetics, greens_greens, data_data, data_greens,
    source, i1, i2):
    """
    Calculates error in the righthand-side terms of
    ||s - d||^2 = s^2 + d^2 - 2sd
    """
    dd = np.sum(data**2)
    print 'error dd:',\
        (dd - data_data[i1])/dd

    ss = np.sum(synthetics**2)
    print 'error ss:',\
        (ss - np.dot(np.dot(greens_greens[i1, i2, :, :], source), source))/ss

    sd = np.sum(synthetics*data)
    print 'error sd:',\
        (sd - np.dot(data_greens[i1, :, i2], source))/sd

    print ''


def get_time_shift(corr, corr_sum, source, indices):
    """ 
    Finds optimal time shift between the given data and synthetics
    generated from the given source
    """
    npts_padding = (len(corr_sum)-1)/2
    ngf = corr.shape[1]

    corr_sum[:] = 0.
    for _i in indices:
        corr_sum += np.dot(source, corr[_i, :, :])

    return corr_sum.argmax() - npts_padding



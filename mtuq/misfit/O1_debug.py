"""
Data misfit module (optimized Python version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
from mtuq.util import iterable
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import get_components
from scipy.signal import fftconvolve


def _corr(v1, v2):
    # fast cross-correlation of unpadded array v1 and padded array v2
    n1, n2 = len(v1), len(v2)

    if n1>2000 or n2-n1>200:
        # for long traces, frequency-domain implementation is usually faster
        return fftconvolve(v1, v2[::-1], 'valid')
    else:
        # for short traces, time-domain implementation is usually faster
        return np.correlate(v1, v2, 'valid')


def _corr_nd1_nd2(data, greens, time_shift_max):
    # correlates 1D and 2D data structures
    corr_all = []

    for d, g in zip(data, greens):

        ncomp = len(g.components)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = _get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[1]

        # array that will hold correlations
        corr = np.zeros((ncomp, ngf, 2*npts_padding+1))

        # the main work starts now
        for _i, component in enumerate(g.components):
            trace = d.select(component=component)[0]

            for _j in range(ngf):
                corr[_i, _j, :] =\
                    _corr(trace.data, array[_i, _j, :])

        corr_all += [corr]

    return corr_all


def _autocorr_nd1(data, time_shift_max):
    # autocorrelates 1D data strucutres (reduces to dot product)
    corr_all = []

    for d in data:
        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        corr = np.zeros(ncomp)

        # the main work starts now
        for _i1, trace in enumerate(d):
            corr[_i1] = np.dot(trace.data, trace.data)

        corr_all += [corr]

    return corr_all


def _autocorr_nd2(greens, time_shift_max):
    # correlates 2D data structures
    corr_all = []

    for g in greens:
        ncomp = len(g.components)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = _get_time_sampling(g)
        npts_padding = int(time_shift_max/dt)

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[1]

        # array that will hold correlations
        corr = np.zeros((ncomp, ngf, ngf, 2*npts_padding+1))

        # the main work starts now
        for _i in range(ncomp):
            for _j1 in range(ngf):
                for _j2 in range(ngf):

                    if _j1<=_j2:
                        # calculate upper triangular elements
                        corr[_i, _j1, _j2, :] = _corr(
                            array[_i, _j1, :]*array[_i, _j2, :],
                            np.pad(np.ones(npts), npts_padding, 'constant'))

                    else:
                        # fill in lower trianglular elements by symmetry
                        corr[_i, _j1, _j2, :] = corr[_i, _j2, _j1, :]

        corr_all += [corr]

    return corr_all


def _corr_init(data, time_shift_max):
    # allocates arrays to hold correlations
    corr_all = []

    for d in data:
        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = _get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        corr_all += [np.zeros(2*npts_padding+1)]

    return corr_all


def _get_L2_norm(greens_greens, data_data, data_greens, source, i1, i2):
    # calculates L2 norm using ||s - d||^2 = s^2 + d^2 - 2sd
    misfit = 0

    # calculates d^2
    misfit += data_data[i1]

    # calculates s^2 using a linear combination of Greens' function
    misfit += np.dot(np.dot(greens_greens[i1, :, :, i2], source), source)

    # calculates sd using a linear combination of Greens' function
    misfit -= 2*np.dot(data_greens[i1, :, i2], source)

    return misfit


def _get_time_shift(corr, corr_sum, source, indices):
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


def _get_time_sampling(stream):
    if len(stream) > 0:
        npts = stream[0].data.size
        dt = stream[0].stats.delta
    else:
        npts = None
        dt = None
    return npts, dt



def misfit(
    data,
    greens,
    sources,
    norm,
    time_shift_groups,
    time_shift_max,
    set_attributes=False,
    verbose=0,
    debug=0):
    """
    Data misfit function (optimized Python version)

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
    # cross-correlations are a major part of the computational work
    #
    corr_sum = _corr_init(data, time_shift_max)
    data_greens = _corr_nd1_nd2(data, greens, time_shift_max)

    if norm in ['L2', 'hybrid']:
        data_data = _autocorr_nd1(data, time_shift_max)
        greens_greens = _autocorr_nd2(greens, time_shift_max)

    #
    # begin iterating over sources
    #
    for _i, source in enumerate(sources):
        for _j, d in enumerate(data):

            components = greens[_j].components
            if not components:
                continue

            if norm in ['L2', 'hybrid']:
                d2 = data_data[_j]
                G2 = greens_greens[_j]
                dG = data_greens[_j]

            if norm in ['L1']:
                s = greens[_j].get_synthetics(source)

            if debug:
                s = greens[_j].get_synthetics(source)

            # time sampling scheme
            npts, dt = _get_time_sampling(d)
            npts_padding = int(time_shift_max/dt)


            #
            # evaluate misfit for all components at given station
            # 
            for group in time_shift_groups:
                # Finds the time-shift between data and synthetics that 
                # yields the maximum cross-correlation value across all
                # components in a given group, subject to time_shift_max 
                # constraint
                _, indices = list_intersect_with_indices(
                    components, group)

                npts_shift = _get_time_shift(
                    data_greens[_j], corr_sum[_j], source, indices)

                time_shift = npts_shift*dt

                # what start and stop indices will correctly shift 
                # synthetics relative to data?
                start = npts_padding - npts_shift
                stop = npts+npts_padding - npts_shift

                for _k in indices:
                    misfit = 0.

                    # sum the resulting residuals
                    if norm=='L1':
                        r = s[_k].data[start:stop] - d[_k].data
                        misfit = np.sum(abs(r))*dt

                    elif norm=='L2':
                        misfit = _get_L2_norm(
                            greens_greens[_j], data_data[_j], data_greens[_j],
                            source, _k, npts_shift+npts_padding)

                        if debug:
                            r = s[_k].data[start:stop] - d[_k].data
                            assert isclose(misfit, np.sum(r**2)*dt)

                    elif norm=='hybrid':
                        misfit = _get_L2_norm(
                            greens_greens[_j], data_data[_j], data_greens[_j],
                            source, _k, npts_shift+npts_padding)**0.5

                        if debug:
                            r = s[_k].data[start:stop] - d[_k].data
                            assert isclose(misfit, (np.sum(r**2)*dt)**0.5)

                    results[_i] += d[_k].weight * misfit

    return results


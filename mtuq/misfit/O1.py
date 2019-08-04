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
    raise NotImplementedError

    # autocorrelates 1D data strucutres
    corr_all = []

    for d in data:

        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = _get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        # array that will hold correlations
        corr = np.zeros((ncomp, npts_padding))

        # the main work starts now
        for _i1, component in enumerate(components):
            trace = d.select(component=component)

            corr[_i1, :] =\
                _corr(trace.data, np.pad(trace.data))

        corr_all += [corr]

    return corr_all


def _autocorr_nd2(greens, time_shift_max):
    # autocorrelates 2D data structures
    corr_all = []

    for g in greens:

        ncomp = len(g.components)
        if ncomp==0:
            continue

        npts, dt = _get_time_sampling(g)
        npts_padding = int(time_shift_max/dt)

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[2]

        # array that will hold correlations
        corr = np.zeros((ncomp, ngf, ngf, npts_padding))

        # the main work starts now
        for _i in range(ncomp):
            for _j1 in range(ngf):
                for _j2 in range(ngf):

                    corr[_i, _j1, _j2, :] =\
                        _corr(trace.data, array[_j1, _j2, :])

        corr_all += [corr]

    return corr_all


def _corr_init(data, time_shift_max):
    # autocorrelates 1D data strucutres
    corr_all = []

    for d in data:

        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = _get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        # array that will hold correlations
        corr_all += [np.zeros(2*npts_padding+1)]

    return corr_all


def _dot2(corr, icomp, npts_shift, source):
    raise NotImplementedError


def _dot3(corr, icomp, npts_shift, source):
    raise NotImplementedError


def _get_time_shift(corr, corr_sum, source, components):
    """ 
    Finds optimal time shift between the given data and synthetics
    generated from the given source
    """
    npts_padding = (len(corr_sum)-1)/2
    ngf = corr.shape[1]

    corr_sum[:] = 0.
    for component in components:
        _i = components.index(component)
        for _j in range(len(source)):
            corr_sum += source[_j] * corr[_i, _j, :]

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
    verbose=0):
    """
    Data misfit function (optimized Python version)

    See ``mtuq.misfit.Misfit`` for more information
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

    if norm in ['L1', 'L2', 'hybrid']:
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

            elif norm in ['L1']:
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
                group, indices = list_intersect_with_indices(
                    components, group)

                #npts_shift = _get_time_shift(
                #    data_greens[_j], corr_sum[_j], source, group)

                npts_shift = greens[_j].get_time_shift(
                    d, source, group, time_shift_max)

                time_shift = npts_shift*dt

                # what start and stop indices will correctly shift 
                # synthetics relative to data?
                start = npts_padding - npts_shift
                stop = npts+npts_padding - npts_shift

                for _k in indices:
                    # sum the resulting residuals
                    if norm=='L1':
                        r = s[_k].data[start:stop] - d[_k].data
                        misfit = np.sum(abs(r))*dt

                    elif norm=='L2':
                        # using ||s - d||^2 = s^2 + d^2 - 2sd
                        misfit = d2 + _dot3(G2, source, _k, npts_shift) -\
                                 2.*_dot2(dG, source, _k, npts_shift)


                    elif norm=='hybrid':
                        # using ||s - d||^2 = s^2 + d^2 - 2sd
                        misfit = (d2 + _dot3(G2, source, _k, npts_shift) -\
                                 2.*_dot2(dG, source, _k, npts_shift))**0.5

                    results[_i] += d[_k].weight * misfit

                    if set_attributes:
                        d[_k].misfit = misfit
                        d[_k].time_shift = -time_shift
                        s[_k].misfit = misfit
                        s[_k].time_shift = +time_shift
                        s[_k].start = start
                        s[_k].stop = stop

    return results


"""
Data misfit module (fast Python/C version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
from mtuq.misfit.O1 import correlate
#from mtuq.misfit import ext2


def misfit(data, greens, sources, norm, groups, time_shift_max,
    verbose=0):
    """
    Data misfit function (fast Python/C version)

    See ``mtuq/misfit/__init__.py`` for more information
    """
    # get time sampling
    Npts, dt = _get_time_sampling(data)
    Ngroups = len(time_shift_groups)

    Npts_max = int(time_shift_max/dt)
    Npts_min = -Npts_max
    #Npts_min = int(time_shift_min/dt)
    #padding = [Npts_min, Npts_max]

    components = data.get_components()
    groups = _get_groups(groups, components)
    mask = _get_mask(data)

    # data arrays
    data = data.as_array()
    greens = greens.as_array()
    sources = sources.as_array()

    # cross-correlation arrays
    data_data = autocorr1(data)
    greens_Greens = autocorr2(greens, padding)
    data_Greens = corr(data, greens, padding)

    return ext2.misfit(
       data, greens, sources, data_data, data_greens, greens_greens, mask,
       groups, Nshift_min, Nshift_max, norm)


def _get_mask(Data):
    raise NotImplementedError
    Mask = np.zeros((Ncomponents, Nstations))


def _get_groups(groups):
    raise NotImplementedError
    Groups = np.zeros(Ncomponents, Ngroups)



#
# optimized cross-correlation functions
#

def _corr_nd1_nd2(data, greens, time_shift_max):
    # correlates 1D and 2D data structures
    corr_all = []

    for d, g in zip(data, greens):

        ncomp = len(g.components)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[1]

        # array that will hold _correlations
        corr = np.zeros((ncomp, ngf, 2*npts_padding+1))

        # the main work starts now
        for _i, component in enumerate(g.components):
            trace = d.select(component=component)[0]

            for _j in range(ngf):
                corr[_i, _j, :] =\
                    correlate(array[_i, _j, :], trace.data)

        corr_all += [corr]

    return corr_all


def _autocorr_nd1(data, time_shift_max):
    # _autocorrelates 1D data strucutres (reduces to dot product)
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
    # _autocorrelates 2D data structures
    corr_all = []

    for g in greens:
        ncomp = len(g.components)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = get_time_sampling(g)
        npts_padding = int(time_shift_max/dt)

        ones = np.pad(np.ones(npts-2*npts_padding), 2*npts_padding, 'constant')

        # array that holds Green's functions
        array = g._array
        ngf = array.shape[1]

        # array that will hold _correlations
        corr = np.zeros((ncomp, 2*npts_padding+1, ngf, ngf))

        # the main work starts now
        for _i in range(ncomp):
            for _j1 in range(ngf):
                for _j2 in range(ngf):

                    if _j1<=_j2:
                        # calculate upper elements
                        corr[_i, :, _j1, _j2] = correlate(
                            array[_i, _j1, :]*array[_i, _j2, :], ones)

                    else:
                        # fill in lower elements by symmetry
                        corr[_i, :, _j1, _j2] = corr[_i, :, _j2, _j1]

        corr_all += [corr]

    return corr_all


def _corr_init(data, time_shift_max):
    # allocates arrays to hold _correlations
    corr_all = []

    for d in data:
        ncomp = len(d)
        if ncomp==0:
            corr_all += [[]]
            continue

        npts, dt = get_time_sampling(d)
        npts_padding = int(time_shift_max/dt)

        corr_all += [np.zeros(2*npts_padding+1)]

    return corr_all




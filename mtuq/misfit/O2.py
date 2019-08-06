"""
Data misfit module (fast Python/C version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
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



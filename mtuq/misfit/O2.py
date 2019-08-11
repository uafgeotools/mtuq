"""
Data misfit module (fast Python/C version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
from mtuq.misfit.O1 import correlate
from mtuq.util.signal import get_components, get_time_sampling
from mtuq.misfit import ext1


def misfit(data, greens, sources, norm, time_shift_groups, time_shift_max,
    verbose=0):
    """
    Data misfit function (fast Python/C version)

    See ``mtuq/misfit/__init__.py`` for more information
    """
    _, dt = get_time_sampling(data[0])
    padding = _get_padding(time_shift_max, dt)

    stations = _get_stations(data)
    components = _get_components(data)

    # boolean arrays
    groups = _get_groups(time_shift_groups, components)
    mask = _get_mask(data, components)

    # data arrays
    data = data.as_array(components)
    greens = _get_greens(greens, stations, components)
    sources = _as_array(sources)

    # correlation arrays
    data_data = _autocorr_nd1(data)
    greens_greens = _autocorr_nd2(greens, padding)
    greens_data = _corr_nd1_nd2(data, greens, padding)

    return ext1.misfit(
       data, greens, sources, data_data, greens_data, greens_greens,
       dt, padding[0], padding[1], verbose)

    #return ext1.misfit(
    #   data, greens, sources, data_data, greens_data, greens_greens,
    #   mask, groups, padding[0], padding[1], norm)


def _get_padding(time_shift_max, dt):
    npts = int(time_shift_max/dt)
    return [npts, npts]


def _get_greens(greens, stations, components):
    Ncomponents = len(components)
    Nstations = len(stations)
    Npts = len(greens[0][0])

    Ngreens = 0
    if greens[0].include_mt:
        Ngreens += 6
    if greens[0].include_force:
        Ngreens += 3

    array = np.zeros((
        Nstations,
        Ncomponents,
        Ngreens,
        Npts,
        ))

    for _i, station in enumerate(stations):
        tensor = greens.select(station=station)[0]

        # populate array elements
        tensor.reset_components(components)

        # fill in array
        array[_i, :, :, :] = tensor._array

    return array


def _get_components(data):
    components = list()
    for stream in data:
        components.extend(get_components(stream))
    components = list(set(components))

    components_sorted = []
    if 'Z' in components:
        components_sorted += ['Z']
    if 'R' in components:
        components_sorted += ['R']
    if 'T' in components:
        components_sorted += ['T']

    return components_sorted


def _get_mask(data, components):
    Ncomponents = len(components)
    Nstations = len(data)

    mask = np.ones((
        Ncomponents, 
        Nstations,
        ))

    for _j in range(Nstations):
        for _i, component in enumerate(components):

            stream = data[_j].select(component=component)

            if len(stream)==0 or\
               np.all(np.isclose(stream[0].data, 0.)):
                mask[_i, _j] = 0.

    return mask


def _get_stations(data):
    stations = []
    for stream in data:
        if len(stream)==0:
            continue
        stations += [stream.station]
    return stations


def _get_groups(groups, components):
    Ncomponents = len(components)
    Ngroups = len(groups)

    array = np.zeros((
        Ngroups, 
        Ncomponents, 
        ))

    for _i, group in enumerate(groups):
        for _j, component in enumerate(components):
            array[_i, _j] = 1

    return array


def _as_array(sources):
    Ngreens = len(sources.get(0))
    Nsources = len(sources)

    array = np.zeros((
        Nsources,
        Ngreens, 
        ))

    for _i, source in enumerate(sources):
        array[_i, :] = source

    return array


#
# optimized cross-correlation functions
#

def _corr_nd1_nd2(data, greens, padding):
    # correlates 1D and 2D data structures
    Ncomponents = greens.shape[1]
    Nstations = greens.shape[0]
    Ngreens = greens.shape[2]

    corr = np.zeros((
        Nstations,
        Ncomponents,
        Ngreens,
        padding[0]+padding[0]+1,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            for _k in range(Ngreens):

                corr[_i, _j, _k, :] = correlate(
                    greens[_i, _j, _k, :], data[_i, _j, :])

    return corr


def _autocorr_nd1(data):
    # autocorrelates 1D data strucutres (reduces to dot product)
    Ncomponents = data.shape[1]
    Nstations = data.shape[0]

    corr = np.zeros((
        Nstations,
        Ncomponents,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            corr[_i, _j] = np.dot(
                data[_i, _j, :], data[_i, _j, :])

    return corr


def _autocorr_nd2(greens, padding):
    # autocorrelates 2D data structures

    Ncomponents = greens.shape[1]
    Nstations = greens.shape[0]
    Ngreens = greens.shape[2]
    Npts = greens.shape[3]

    ones = np.pad(np.ones(Npts), 
        (padding[0], padding[1]), 'constant')

    corr = np.zeros((
        Nstations,
        Ncomponents, 
        padding[0]+padding[0]+1, 
        Ngreens, 
        Ngreens,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            for _k1 in range(Ngreens):
                for _k2 in range(Ngreens):

                    if _k1<=_k2:
                        corr[_i, _j, :, _k1, _k2] = correlate(
                            greens[_i, _j, _k1, :]*greens[_i, _j, _k2, :],
                            ones)

                    else:
                        corr[_i, _j, :, _k1, _k2] = corr[_i, _j, :, _k2, _k1]

    return corr



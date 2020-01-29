"""
Data misfit module (fast Python/C version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
import time
from copy import deepcopy
from mtuq.misfit.fast1 import correlate
from mtuq.util.lune import to_mij, to_xyz
from mtuq.util.signal import get_components, get_time_sampling
from mtuq.misfit import c_ext_L2


def misfit(data, greens, sources, norm, time_shift_groups,
    time_shift_min, time_shift_max, verbose=0):
    """
    Data misfit function (fast Python/C version)

    See ``mtuq/misfit/__init__.py`` for more information
    """
    nt, dt = _get_time_sampling(data)
    padding = _get_padding(time_shift_min, time_shift_max, dt)

    stations = _get_stations(data)
    components = _get_components(data)

    # boolean arrays
    groups = _get_groups(time_shift_groups, components)
    mask = _get_mask(data, stations, components)

    # data arrays
    data = _get_data(data, stations, components, nt)
    greens = _get_greens(greens, stations, components)
    sources = _as_array(sources)

    # correlation arrays
    data_data = _autocorr_1(data)
    greens_greens = _autocorr_2(greens, padding)
    greens_data = _corr_1_2(data, greens, padding)

    if norm=='hybrid':
        hybrid_norm = 1
    else:
        hybrid_norm = 0

    start_time = time.time()

    if norm in ['L2', 'hybrid']:
        results = c_ext_L2.misfit(
           data_data, greens_data, greens_greens, sources, groups, mask,
           hybrid_norm, dt, padding[0], padding[1], int(verbose))

    else:
        raise NotImplementedError

    if verbose:
      print('  Elapsed time (C extension) (s): %f' % \
          (time.time() - start_time))

    return results


def _get_time_sampling(dataset):
    for stream in dataset:
        if len(stream) > 0:
            return get_time_sampling(stream)


def _get_padding(time_shift_min, time_shift_max, dt):
    padding_left = int(+time_shift_max/dt)
    padding_right = int(-time_shift_min/dt)
    return [padding_left, padding_right]


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
        tensor = greens.select(station)[0]

        # fill in array
        tensor._set_components(components)
        array[_i, :, :, :] = tensor._array

    return array


def _get_data(data, stations, components, nt):
    #Collects numeric trace data from all streams as a single NumPy array

    #Compared with iterating over streams and traces, provides a potentially
    #faster way of accessing numeric trace data

    #.. warning::

    #    Requires that all streams have the same time discretization
    #    (or else an error is raised)

    ns = len(stations)
    nc = len(components)
    array = np.zeros((ns, nc, nt))

    for _i, station in enumerate(stations):
        stream = data.select(station)[0]
        for _j, component in enumerate(components):
            try:
                trace = stream.select(component=component)[0]
            except:
                continue
            array[_i, _j, :] = trace.data

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


def _get_mask(data, stations, components):
    Ncomponents = len(components)
    Nstations = len(stations)

    mask = np.ones((
        Nstations,
        Ncomponents, 
        ))

    for _i, station in enumerate(stations):
        for _j, component in enumerate(components):

            stream = data.select(station)[0]
            stream = stream.select(component=component)

            if len(stream)==0:
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
            if component in group:
                array[_i, _j] = 1

    return array


def _as_array(sources):
    array = sources.as_array(callback=None)

    if array.shape[1]==6:
        # convert moment tensors to Mij parameterization (up,south,east)
        return np.ascontiguousarray(to_mij(
            array[:,0], array[:,1], array[:,2],  # rho, v, w
            array[:,3], array[:,4], array[:,5])) # kappa, sigma, h

    elif array.shape[1]==3:
        # convert forces to F1,F2,F3 parameterization (1:up,2:south,3:east)
        return np.ascontiguousarray(to_xyz(
            array[:,0], array[:,1], array[:,2])) # F0 (radius), theta (azimuth), phi (colatitude)


#
# optimized cross-correlation functions
#

def _corr_1_2(data, greens, padding):
    # correlates 1D and 2D data structures
    Ncomponents = greens.shape[1]
    Nstations = greens.shape[0]
    Ngreens = greens.shape[2]

    corr = np.zeros((
        Nstations,
        Ncomponents,
        Ngreens,
        padding[0]+padding[1]+1,
        ))

    for _i in range(Nstations):
        for _j in range(Ncomponents):
            for _k in range(Ngreens):

                corr[_i, _j, _k, :] = correlate(
                    greens[_i, _j, _k, :], data[_i, _j, :])

    return corr


def _autocorr_1(data):
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


def _autocorr_2(greens, padding):
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
        padding[0]+padding[1]+1, 
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



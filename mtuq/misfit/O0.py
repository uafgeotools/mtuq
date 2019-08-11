"""
Data misfit module (non-optimized pure Python version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np

from mtuq.util import iterable
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import get_components


def misfit(data, greens, sources, norm, time_shift_groups, time_shift_max,
    set_attributes=False,verbose=0):
    """
    Data misfit function (non-optimized pure Python version)

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
    # begin iterating over sources
    #
    for _i, source in enumerate(sources):
        for _j, d in enumerate(data):

            components = greens[_j].components
            if not components:
                continue

            # generate synthetics
            s = greens[_j].get_synthetics(source)

            # time sampling scheme
            npts = d[0].data.size
            dt = d[0].stats.delta
            npts_padding = int(time_shift_max/dt)

            # array to hold cross correlations
            corr = np.zeros(2*npts_padding+1)


            #
            # evaluate misfit for all components at given station
            # 
            for group in time_shift_groups:
                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all components in 
                # a given group, subject to time_shift_max constraint
                _, indices = list_intersect_with_indices(
                    components, group)

                for _k in indices:
                    corr += np.correlate(d[_k].data, s[_k].data, 'valid')

                npts_shift = corr.argmax()-npts_padding
                time_shift = npts_shift*dt

                # what start and stop indices will correctly shift synthetics
                # relative to data?
                start = npts_padding-npts_shift
                stop = npts+npts_padding-npts_shift

                for _k in indices:
                    # substract data from shifted synthetics
                    r = s[_k].data[start:stop] - d[_k].data

                    # sum the resulting residuals
                    if norm=='L1':
                        misfit = np.sum(abs(r))*dt

                    elif norm=='L2':
                        misfit = np.sum(r**2)*dt

                    elif norm=='hybrid':
                        misfit = np.sqrt(np.sum(r**2))*dt

                    results[_i] += d[_k].weight * misfit

                    if set_attributes:
                        d[_k].misfit = misfit
                        d[_k].time_shift = -time_shift
                        s[_k].misfit = misfit
                        s[_k].time_shift = +time_shift
                        s[_k].start = start
                        s[_k].stop = stop

    return results



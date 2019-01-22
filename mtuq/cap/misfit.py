
from collections import defaultdict
from math import ceil, floor
from scipy.signal import fftconvolve
from mtuq.util.math import isclose, list_intersect_with_indices
import numpy as np
import warnings


class Misfit(object):
    """ CAP-style data misfit function

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = cap_misfit(**parameters)
        misfit = function_handle(data, greens, mt)

    In the first step, the user supplies a list of parameters, including
    the order of the norm applied to the residuals, whether or not to use
    polarity information, and various tuning parameters (see below for detailed
    descriptions.) In the second step, the user supplies data, greens function,
    and source object. Synthetics are then generated and compared with data, and
    a corresponding misfit value returned.
    """

    def __init__(self,
        norm_order=1,
        polarity_weight=0.,
        time_shift_groups=['ZRT'],
        time_shift_max=0.,
        ):
        """ Checks misfit parameters

        time_shift_groups
            ['ZRT'] locks time-shift across all three components
            ['ZR','T'] locks vertical and radial components only
            ['Z','R','T'] allows time shifts to vary freely between components

        """
        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T']

        # what norm should we apply to the residuals?
        self.order = norm_order

        # maximum cross-correlation lag (seconds)
        self.time_shift_max = time_shift_max

        # should we allow time shifts to vary from component to component?
        self.time_shift_groups = time_shift_groups

        # should we include polarities in misfit?
        self.polarity_weight = polarity_weight

        # keeps track of what components are available in each stream
        self._components = defaultdict(list)


    def __call__(self, data, greens, mt):
        """ CAP-style misfit calculation
        """ 
        p = self.order

        sum_misfit = 0.
        for _i, d in enumerate(data):
            # what components are in stream d?
            if _i not in self._components:
                for trace in d:
                    self._components[_i] += [trace.stats.channel[-1].upper()]
                greens[_i].components = self._components[_i]

            components = self._components[_i]
            if not components:
                continue

            # generate synthetics
            s = greens[_i].get_synthetics(mt)

            # time sampling scheme
            npts = d[0].data.size
            dt = d[0].stats.delta
            npts_padding = int(self.time_shift_max/dt)


            #
            # PART 1: CAP-style waveform-difference misfit calculation, with
            #     time-shift corrections
            #
             
            for group in self.time_shift_groups:
                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all components in 
                # in a given group, subject to time_shift_max constraint

                # what components are in stream d?
                group, indices = list_intersect_with_indices(components, group)

                # what time-shift yields the maximum cross-correlation value?
                offset = greens[_i].get_time_shift(d, mt, group, self.time_shift_max)
                time_shift = offset*dt

                # what start and stop indices will correctly shift synthetics 
                # relative to data?
                start = npts_padding-offset
                stop = npts+npts_padding-offset

                for _j in indices:
                    s[_j].time_shift = time_shift
                    s[_j].time_shift_group = group
                    s[_j].start = start
                    s[_j].stop = stop
                    
                    # substract data from shifted synthetics
                    r = s[_j].data[start:stop] - d[_j].data

                    # sum the resulting residuals
                    d[_j].sum_residuals = np.sum(np.abs(r)**p)*dt
                    sum_misfit += d[_j].weight * d[_j].sum_residuals



            #
            # PART 2: CAP-style polarity calculation
            #

            if self.polarity_weight > 0.:
                raise NotImplementedError


        return sum_misfit**(1./p)



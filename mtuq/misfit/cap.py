
from collections import defaultdict
from math import ceil, floor
from scipy.signal import fftconvolve
from mtuq.util.math import isclose
import numpy as np


class misfit(object):
    """ 
    CAP-style data misfit function

    Evaluating misfit is a two-step procedure:
        1) function_handle = cap_misfit(**parameters)
        2) misfit = function_handle(data, synthetics)

    In the first step, the user supplies a list of parameters, including
    the order of the norm applied to the residuals, whether or not to use
    polarity information, and various tuning parameters (see below for detailed
    descriptions.) In the second step, the user supplies data and synthetics 
    and gets back the corresponding misfit value.
    """

    def __init__(self,
        norm_order=2,
        polarity_weight=0.,
        time_shift_groups=['ZRT'],
        time_shift_max=None,
        ):
        """ Checks misfit parameters

        time_shift_groups
            ['ZRT'] locks time-shift across all three components
            ['ZR','T'] locks vertical and radial components only
            ['Z','R','T'] allows time shifts to vary freely between components

        """
        # what norm should we apply to the residuals?
        self.order = norm_order

        # should we allow time shifts to vary from component to component?
        self.time_shift_groups = time_shift_groups

        # should we include polarities in misfit?
        self.polarity_weight = polarity_weight


    def __call__(self, data, synthetics, mode=1):
        """ CAP-style misfit calculation
        """ 
        p = self.order

        sum_misfit = 0.
        for d, s in zip(data, synthetics):
            # time sampling scheme
            nt = d[0].data.size
            dt = d[0].stats.delta

            # by how many samples are synthetics padded relative to data?
            npad = (s[0].data.size-nt)/2

            #
            # PART 1: CAP-style waveform difference misfit
            #
             
            for group in self.time_shift_groups:
                _indices = []

                # array to hold cross-correlation result
                result = np.zeros(2*npad+1)

                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all components in 
                # in a given group, subject to the time_shift_max constraint
                for _i in range(len(d)):
                    if d[_i].weight == 0.:
                        # ignore components with zero weight
                        continue

                    component = d[_i].stats.channel[-1].upper()
                    if component not in group:
                        continue
                    _indices += [_i]

                    if mode==1:
                        # scipy frequency-domain implementation
                        result += fftconvolve(
                            d[_i].data, s[_i].data[::-1], 'valid')

                    elif mode==2:
                        # numpy time-domain implemenation
                        result += np.correlate(
                            d[_i].data, s[_i].data, 'valid')

                # what time-shift yields the maximum cross-correlation value?
                offset = result.argmax()
                time_shift = (offset-npad)*dt

                # sums waveform difference residuals
                for _i in _indices:
                    s[_i].offset = offset
                    s[_i].time_shift = time_shift
                    
                    # substract data from shifted synthetics
                    r = s[_i].data[offset:offset+nt] - d[_i].data

                    # sum the resulting residuals
                    sum_misfit += d[_i].weight * np.sum(r**p)*dt


            #
            # PART 3: CAP-style polarity misfit
            #

            if self.polarity_weight > 0.:
                raise NotImplementedError


        return sum_misfit**(1./p)



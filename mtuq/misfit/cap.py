
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
        time_shift_max=0.,
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
        self.time_shift_max = time_shift_max

        # should we allow time shifts to vary from component to component?
        self.time_shift_groups = time_shift_groups

        # should we include polarities in misfit?
        self.polarity_weight = polarity_weight


    def __call__(self, data, synthetics):
        """ CAP-style misfit calculation
        """ 
        p = self.order

        sum_misfit = 0.
        for d, s in zip(data, synthetics):
            # time sampling scheme
            npts = d[0].data.size
            dt = d[0].stats.delta
            npts_padding = int(round(self.time_shift_max/dt))


            #
            # PART 1: What method to compute time-shift corrections?
            #

            if not hasattr(d, 'time_shift_mode'):
                npts_dat = npts
                npts_syn = s[0].data.size

                if (npts_syn-npts_dat)/2 != npts_padding:
                   raise Exception("Mismatch between npts_syn and npts_dat: "
                       "please double check that process_data.padding_length "
                       "matches misfit.time_shift_max")

                # Chooses whether to work in the time or frequency domain based 
                # on length of traces and maximum allowable lag
                if npts_padding==0:
                    d.time_shift_mode = 0
                elif npts > 500 or npts_padding > 50:
                    # for all but the shortest traces, frequency domain
                    # implementation is usually faster
                    d.time_shift_mode = 1
                else:
                    d.time_shift_mode = 2


            #
            # PART 2: CAP-style waveform-difference misfit calculation
            #
             
            for group in self.time_shift_groups:
                _indices = []

                # array to hold cross-correlation result
                result = np.zeros(2*npts_padding+1)

                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all components in 
                # in a given group, subject to time_shift_max constraint
                for _i in range(len(d)):
                    if d[_i].weight == 0.:
                        # ignore components with zero weight
                        continue

                    component = d[_i].stats.channel[-1].upper()
                    if component not in group:
                        continue
                    _indices += [_i]

                    if d.time_shift_mode==0:
                        pass

                    elif d.time_shift_mode==1:
                        # scipy frequency-domain implementation
                        result += fftconvolve(
                            d[_i].data, s[_i].data[::-1], 'valid')

                    elif d.time_shift_mode==2:
                        # numpy time-domain implemenation
                        result += np.correlate(
                            d[_i].data, s[_i].data, 'valid')


                # what time-shift yields the maximum cross-correlation value?
                argmax = result.argmax()
                time_shift = (argmax-npts_padding)*dt

                # sums waveform difference residuals
                for _i in _indices:
                    s[_i].argmax = argmax
                    s[_i].time_shift = time_shift
                    
                    # substract data from shifted synthetics
                    r = s[_i].data[argmax:argmax+npts] - d[_i].data

                    # sum the resulting residuals
                    sum_misfit += d[_i].weight * np.sum(r**p)*dt



            #
            # PART 3: CAP-style polarity misfit calculation
            #

            if self.polarity_weight > 0.:
                raise NotImplementedError


        return sum_misfit**(1./p)



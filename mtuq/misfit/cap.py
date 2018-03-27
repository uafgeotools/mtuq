
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
    the order of the norm applied to residuals, whether or not to use
    polarity information, and various tuning parameters (see below for detailed
    descriptions.) In the second step, the user supplies data and synthetics 
    and gets back the corresponding misfit value.
    """

    def __init__(self,
        norm_order=2,
        polarity_weight=0.,
        time_shift_window_length=None,
        time_shift_max=None,
        ):
        """ Checks misfit parameters
        """
        # what norm should we apply to residuals?
        self.order = norm_order

        # what portion of data and synthetics should we consider?
        self.time_shift_window_length = time_shift_window_length

        # what maximum offset should we stop at when cross correlating?
        self.time_shift_max = time_shift_max

        # should we include polarities in misfit?
        self.polarity_weight = polarity_weight


    def __call__(self, data, synthetics, mode=1):
        """ CAP-style misfit calculation
        """ 
        p = self.order

        sum_misfit = 0.
        for d, s in zip(data, synthetics):
            # time sampling scheme
            nt = d[0].stats.npts
            dt = d[0].stats.delta


            #
            # PART 1: Calculate CAP-style time shifts
            #
             
            # Finds the time-shift between data and synthetics that yields the
            # maximum cross-correlation value across all components, subject to 
            # window length and time_shift_max constraints.  The result is a single 
            # time-shift which is the same for all components at a given station.

            if self.time_shift_window_length:
                n1 = min(round(self.time_shift_window_length/dt), nt)
            else:
                n1 = nt

            if self.time_shift_max:
                n2 = min(round(self.time_shift_max/dt), n1/2)
            else:
                n2 = n1/2

            # what portion of data and synthetics are we considering?
            if n1 < nt:
               if not hasattr(d[_i], 'arrival'):
                   raise Exception
               t0 = d[_i].arrival-d[_i].stats.starttime
               it1 = min(int(round((t0/dt-n1/2))), 0)
               it2 = it1 + n1/2
            else:
               it1 = 0
               it2 = nt

            # perform cross correlations
            result = np.zeros(n1)
            for _i in range(len(d)):
                if d[_i].weight == 0.:
                    # ignore components with zero weight
                    continue

                elif mode==1:
                    # scipy frequency-domain implementation (usually much faster)
                    result += fftconvolve(
                        d[_i].data[it1:it2], s[_i].data[it1:it2][::-1], 'same')

                elif mode==2:
                    # numpy time-domain implemenation
                    result += np.correlate(
                        d[_i].data[it1:it2], s[_i].data[it1:it2], 'same')

                elif mode==3:
                    # mtuq time-domain implementation
                    raise NotImplementedError


            # what time-shift yields the maximum cross-correlation value?
            if n2 < n1/2:
                it1 = int(n1/2 - n2)
                it2 = int(n1/2 + n2)
            else:
                it1 = 0
                it2 = n1

            d.time_shift = (result[it1:it2].argmax()+it2-it1+1)*dt


            #
            # PART 2: waveform difference calculation
            #

            # Sums waveform difference residuals for all component

            for _i in range(len(d)):
                if isclose(d[_i].weight, 0.):
                    # ignore components with zero weight
                    continue

                # shift data and synthetics by same amount but in opposite
                # directions
                it = int(round(d.time_shift/(2.*dt)))

                # substract shifted data from shifted synthetics
                if it == 0:
                    r = s[_i] - d[_i]
                elif it < 0:
                    r = s[_i][it:] - d[_i][:-it]
                elif it > 0:
                    r = s[_i][:-it] - d[_i][it:]

                # sum the resulting residuals
                sum_misfit += d[_i].weight * np.sum(r**p)*dt


            #
            # PART 3: optionally, include polarity information
            #

            if self.polarity_weight > 0.:
                raise NotImplementedError


        return sum_misfit**(1./p)



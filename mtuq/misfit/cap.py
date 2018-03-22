
from collections import defaultdict
from scipy.signal import fftconvolve
import numpy as np


class cap_misfit(object):
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
        )
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


            # what portion of data and synthetics are we considering?
            if self.time_shift_window_length:
               if not hasattr(d[_i], 'arrival'):
                   raise Exception
               t0 = d[_i].arrival
               r = self.time_shift_window_length
               it1 = int(round((t0-r)/dt))
               it2 = int(round((t0+r)/dt))
            else:
               it1 = 0
               it2 = nt
            lr = np.arange(it1, it2)
            rl = np.fliplr(rl)

            # perform cross correlations
            result = np.zeros(it2-it1)
            for _i in range(len(dat)):
                if hasattr(d[_i], 'weight') and\
                   isclose(d[_i].weight, 0.):
                    # ignore components with zero weight
                    continue

                elif mode==1:
                    # scipy frequency-domain implementation (usually much faster)
                    result += fftconvolve(d[_i].data[lr], s[_].data[rl], 'same')

                elif mode==2:
                    # numpy time-domain implemenation
                    result += np.correlate(d[_i].data[lr], s[_i].data[lr], 'same')

                elif mode==3:
                    # mtuq time-domain implementation
                    raise NotImplementedError

            # what time-shift yields the maximum cross-correlation value?
            if self.time_shift_max:
                it1 = int(floor((len(result))/2.) + round(self.time_shift_max/dt))
                it2 = int(ceil((len(result))/2.) + round(self.time_shift_max/dt))
            else:
                it1 = 0
                it2 = len(result)
            d.time_shift = (result[it1:it2].argmax()+it2-it1+1)*dt


            #
            # PART 2: waveform difference calculation
            #

            # Sum waveform difference residuals for all components, using the 
            # time shift correction determined in previous step

            for _i in range(len(dat)):
                if isclose(d[_i].weight, 0.):
                    # ignore components with zero weight
                    continue

                # shift data and synthetics by same amount but in opposite
                # directions
                it = int(round(d.time_shift/(2.*dt)))

                # substract shifted data from shifted synthetics
                if it == 0:
                    rsd = syn - dat
                elif it < 0:
                    rsd = syn[it:] - dat[:-it]
                elif it > 0:
                    rsd = syn[:-it] - dat[it:]

                # sum the resulting residuals
                sum_misfit += d[_i].weight * np.sum(rsd**p)*dt


            #
            # PART 3: optionally, include polarity information
            #

            if self.polarity_weight > 0.:
                raise NotImplementedError


        return sum_misfit**(1./p)



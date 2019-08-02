
import numpy as np

from scipy.signal import fftconvolve
from mtuq.util.math import isclose, list_intersect_with_indices



class Misfit(object):
    """ Non-optimized Pure Python misfit function

    Evaluates misfit between data and synthetics using time shifts followed by
    waveform differences. This approach, due to ZhaoHelmberger1994 and 
    ZhuHelmberger1996, has become widely used in regional seismology.  See 
    `docs/` and references therein for a more thorough mathematical description.

    .. Usage:

        Evaluating misfit is a two-step procedure. In the first step, the user
        supplies a list of parameters, including the norm applied to the 
        waveform differences (see `__init__` method for detailed argument 
        descriptions). 

        In the second step, the user supplies data, Green's functions, and an
        iterable sources grid (see `__call__` method  for detailed argument 
        descriptions). Synthetics are then  generated and compared with data, 
        and a numpy array is returned with the same length as `sources`.

        Putting both steps together, here is an example of how a misfit 
        function might be created and used:

        .. code::
            function_handle = Misfit(norm='L2')
            array = function_handle(data, greens, sources)


    .. Context:

        Misfit evaluation is the most complex and computationally expensive 
        task performed by this software package. Compared with
        ZhuHelmberger1996's original C software, Python offers major advantages
        for managing this complexity.  Python object-oriented programming makes 
        it possible to offer three different implementations:

        - a readable Pure Python version (`mtuq.misfit.O0.Misfit`)

        - an optimized Pure Python version (`mtuq.misfit.O1.Misfit`)

        - an optimized Python/C++ version (`mtuq.misfit.O2`.Misfit)


        While providing the same input argument syntax, these three versions
        differ significantly in terms of performance:

        - O0 provides a reference for understanding what the code is actually
          doing and for checking the correctness of the optimized O1 and O2 
          results.

        - O1 is an optimized Pure Python implementation which provides 
          significant computational savings when `len(sources)` > 100. This
          version is the closest to ZhuHelmberger1996's original C software,
          and is also the default version accessible by `mtuq.misfit.Misfit`

        - O2 is an optimized Python/C++ implementation, in which a Python 
          wrapper is used to combine obspy Traces into multidimensional arrays.
          These arrays are passed to a C++ extension module, which does the
          main computational work. This version requires that all obspy Traces
          have the same time discretization, which is not a requirement in the
          other two versions.
          

    """

    def __init__(self,
        norm='hybrid',
        polarity_weight=0.,
        time_shift_groups=['ZRT'],
        time_shift_max=0.,
        optimization=1,
        ):
        """ Checks misfit parameters

        norm
            L2: conventional waveform difference
                \Sigma \int (s - d)^2

            L1: more most robust against outliers but slower to compute
                \Sigma \int |s - d|

            hybrid: robust like L1 but fast like L2
                \Sigma \sqrt{ \int (s - d)^2}


        time_shift_groups
            ['ZRT']: locks time-shift across all three components

            ['ZR','T']: locks vertical and radial components only

            ['Z','R','T']: allows time shifts to vary freely between components

        """
        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T']

        # what norm should we apply to the residuals?
         if norm in ['L1', 'L2']:
            pass

         elif type(norm) in [str, unicode]:
            norm = norm.lower()
            assert norm in ['hybrid']

         elif type(norm) in [int, float]:
            self.norm = float(norm)

         else:
            raise ValueError("Bad keyword argument: norm")

        # maximum cross-correlation lag (seconds)
        self.time_shift_max = time_shift_max

        # should we allow time shifts to vary from component to component?
        self.time_shift_groups = time_shift_groups

        # should we include polarities in misfit?
        self.polarity_weight = polarity_weight


    def __call__(self, data, greens, mt):
        """ CAP-style misfit calculation
        """ 
        p = self.norm_order

        sum_misfit = 0.
        for _i, d in enumerate(data):
            try:
                components = greens[_i].components
            except:
                # what components are in stream d?
                components = []
                for trace in d:
                    components += [trace.stats.channel[-1].upper()]
                greens[_i].initialize(components)

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
                    if self.norm in ['L1']:
                        s[_j].misfit = np.sum(abs(r))*dt

                    elif self.norm in ['L2']:
                        s[_j].misfit = np.sum(r**2)*dt

                    elif self.norm in ['Hybrid', 'hybrid']:
                        s[_j].misfit = np.sqrt(np.sum(r**2)*dt)

                    elif type(self.norm)==float:
                        s[_j].misfit = np.sum(np.abs(r)**self.norm)*dt

                    sum_misfit += d[_j].weight * s[_j].misfit


            #
            # PART 2: CAP-style polarity calculation
            #

            if self.polarity_weight > 0.:
                raise NotImplementedError


        return sum_misfit



import numpy as np

from mtuq.util import iterable
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import get_components



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
        time_shift_groups=['ZRT'],
        time_shift_max=0.,
        ):
        """ Checks misfit parameters

        norm
            L2: conventional waveform difference
                `\int (s - d)^2`

            L1: robust against outliers but slower than L2
                `\int |s - d|`

            hybrid: robust like L1 and fast like L2
                `\sqrt{ \int (s - d)^2 }`


        time_shift_groups
            ['ZRT']: locks time-shift across all three components

            ['ZR','T']: locks vertical and radial components only

            ['Z','R','T']: allows time shifts to vary freely between components


        time_shift_max
            maximum allowed cross-correlation lag (seconds)

        """
        if norm.lower()=='hybrid':
            self.norm = norm.lower()

        assert norm in ['L1', 'L2', 'hybrid'],\
            ValueError("Bad input argument")

        assert time_shift_max > 0.,\
            ValueError("Bad input argument")

        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T'],\
                    ValueError("Bad input argument")

        self.norm = norm
        self.time_shift_max = time_shift_max
        self.time_shift_groups = time_shift_groups


    def __call__(self, data, greens, sources, set_attributes=False):
        """ CAP-style misfit calculation
        """ 
        sources = iterable(sources)
        results = np.zeros(len(sources))

        # initialize Green's function machinery
        for _j, d in enumerate(data):
            if not hasattr(greens[_j], 'components'):
                greens[_j].initialize(get_components(d))

        #
        # begin loop over sources
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
                npts_padding = int(self.time_shift_max/dt)


                #
                # evaluate misfit for a given source, station pair
                # 
                for group in self.time_shift_groups:
                    # Finds the time-shift between data and synthetics that 
                    # yields the maximum cross-correlation value across all
                    # components in a given group, subject to time_shift_max 
                    # constraint

                    # what components are in stream d?
                    group, indices = list_intersect_with_indices(
                        components, group)

                    # what time-shift yields the maximum cross-correlation?
                    offset = greens[_j].get_time_shift(
                        d, source, group, self.time_shift_max)

                    time_shift = offset*dt

                    # what start and stop indices will correctly shift 
                    # synthetics relative to data?
                    start = npts_padding-offset
                    stop = npts+npts_padding-offset

                    for _k in indices:
                        # substract data from shifted synthetics
                        r = s[_k].data[start:stop] - d[_k].data

                        # sum the resulting residuals
                        if self.norm=='L1':
                            misfit = np.sum(abs(r))*dt

                        elif self.norm=='L2':
                            misfit = np.sum(r**2)*dt

                        elif self.norm=='hybrid':
                            misfit = np.sqrt(np.sum(r**2)*dt)

                        results[_i] += d[_k].weight * misfit

                        if set_attributes:
                            d[_k].misfit = misfit
                            d[_k].time_shift = -time_shift
                            s[_k].misfit = misfit
                            s[_k].time_shift = +time_shift
                            s[_k].start = start
                            s[_k].stop = stop

        return results


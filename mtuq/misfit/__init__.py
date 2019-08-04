
import numpy as np

from mtuq.misfit import O0, O1
from mtuq.util import iterable
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import get_components



class Misfit(object):
    """ Data misfit function

    Evaluates misfit between data and synthetics using time shifts followed by
    waveform differences. This approach, due to ZhaoHelmberger1994 and 
    ZhuHelmberger1996, has become widely used in regional seismology.  See 
    `docs/` and references therein for a more thorough mathematical description.

    .. Usage:

        Evaluating misfit is a two-step procedure:

        .. code::
            function_handle = Misfit(**parameters)
            array = function_handle(data, greens, sources)

        In the first step, the user supplies parameters such as the type of
        norm to apply the residuals (see `__init__` method for detailed 
        parameter descriptions). 

        In the second step, the user supplies data, Green's functions, and 
        sources (see `__call__` method  for detailed input argument 
        descriptions). Synthetics are then generated and compared with data, 
        and a numpy array is returned with the same length as ``sources``.


    .. Optimization Levels:

        Misfit evaluation is the most complex and computationally expensive 
        task performed by this software package. Compared with
        ZhuHelmberger1996's original C software, Python offers major advantages
        for managing this complexity.  Python object-oriented programming makes 
        it possible to offer three different implementations:

        - a readable pure Python version (`mtuq.misfit.O0.Misfit`)

        - an optimized pure Python version (`mtuq.misfit.O1.Misfit`)

        - an optimized Python/C++ version (`mtuq.misfit.O2`.Misfit)


        While providing the same input argument syntax, these three versions
        differ significantly in terms of performance:

        - O0 provides a reference for understanding what the code is actually
          doing and for checking the correctness of the optimized O1 and O2 
          results.

        - O1 is an optimized pure Python implementation which provides 
          significant computational savings when `len(sources)` > 100. This
          version is the closest to ZhuHelmberger1996's original C software.

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
        time_shift_max=1.,
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
        if norm.lower() in ['hybrid', 'hybrid_l1_l2']:
            norm = 'hybrid'

        assert norm in ['L1', 'L2', 'hybrid'],\
            ValueError("Bad input argument")

        assert time_shift_max >= 0.,\
            ValueError("Bad input argument")

        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T'],\
                    ValueError("Bad input argument")

        self.norm = norm
        self.time_shift_max = time_shift_max
        self.time_shift_groups = time_shift_groups
        self.verbose = 0


    def __call__(self, data, greens, sources, optimization_level=1, 
        set_attributes=False):

        if optimization_level==0 or set_attributes:
            return O0.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_max, set_attributes, self.verbose)

        if optimization_level==1:
            return O1.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_max, set_attributes, self.verbose)

        if optimization_level==2:
            raise NotImplementedError


            

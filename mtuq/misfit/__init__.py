
import numpy as np
import warnings

from mtuq.misfit import simple, fast1#, fast2
from mtuq.util import iterable
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import get_components



class Misfit(object):
    """ Data misfit function

    Evaluates misfit between data and synthetics using time shifts followed by
    waveform differences. This approach, due to `ZhaoHelmberger1994` and 
    `ZhuHelmberger1996`, has become widely used in regional seismology.  See 
    `docs/` and `docs/references` for more information.

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = Misfit(**parameters)
        results = function_handle(data, greens, sources)

    In the first step, the user supplies parameters such as the type of norm
    (see below for detailed argument descriptions).

    In the second step, the user supplies data, Green's functions, and sources.
    Synthetics are then generated and compared with data, and an array of 
    misfit values is returned of the same length as `sources`.



    .. rubric :: Parameters

    ``norm`` (`str`)

    - ``'L2'``: conventional L2 norm (fast)
    ..  r1**2 + r1**2 + ...

    - ``'L1'``: conventional L1 norm (slow)
    ..  \|r1\| + \|r2\| + ...

    - ``'hybrid'``: hybrid L1-L2 norm (much faster than L1 but still robust)
    ..  (r11**2 + r12**2 + ...)**0.5 + (r21**2 + r22**2 + ...)**0.5 + ...


    ``time_shift_groups`` (`list`)

    - ``['ZRT']``: forces all three components to have the same time shift

    - ``['ZR','T'``]: forces vertical and radial components to have the same
      time shift, while transverse time shift is allowed to vary independently

    - ``['Z','R','T']``: allows time shifts of all three components to vary
      independently

    ``time_shift_min`` (`float`): minimum allowable time shift (s)

    ``time_shift_max`` (`float`): maximum allowable time shift (s)



    .. rubric:: Optimization Levels

    Misfit evaluation is the most complex and computationally expensive task
    performed by this software package. Python object-oriented programming
    makes it possible to offer three different implementations:

    - a readable pure Python version (``mtuq.misfit.simple``)

    - a fast pure Python version (``mtuq.misfit.fast1``)

    - a very fast Python/C++ version (``mtuq.misfit.fast2``)


    While exactly the same in terms of input argument syntax, these three 
    versions differ in terms of performance:

    - ``simple`` provides a reference for understanding what the code is doing
      and for checking the correctness of the fast implementations

    - ``fast1`` is an optimized pure Python implementation which provides 
      significant computational savings when `len(sources)` > 100. This
      version is the closest to `ZhuHelmberger1996`'s original C software.

    - ``fast2`` is an optimized Python/C++ implementation, in which a Python 
      wrapper is used to combine obspy Traces into multidimensional arrays.
      These arrays are passed to a C++ extension module, which does the
      main computational work. Unlike the other two versions, this 
      implementation requires that all obspy Traces have the same time
      discretization
      

    """

    def __init__(self,
        norm='hybrid',
        time_shift_groups=['ZRT'],
        time_shift_min=0.,
        time_shift_max=0.,
        ):
        """ Function handle constructor
        """
        # L2 norm 
        # r1**2 + r1**2 + ... 

        # L1 norm
        # |r1| + |r2| + ...

        # hybrid norm
        # (r11**2 + r12**2 + ...)**0.5 + (r21**2 + r22**2 + ...)**0.5 + ...

        if norm.lower()=='hybrid':
            norm = 'hybrid'

        assert norm in ['L1', 'L2', 'hybrid'],\
            ValueError("Bad input argument")

        assert time_shift_max >= 0.,\
            ValueError("Bad input argument")

        if norm=='L1':
            warnings.warn(
                "Consider using norm='hybrid', which is much faster than "
                "norm='L1' but still robust against outliers.")

        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T'],\
                    ValueError("Bad input argument")

        self.norm = norm
        self.time_shift_min = time_shift_min
        self.time_shift_max = time_shift_max
        self.time_shift_groups = time_shift_groups


    def __call__(self, data, greens, sources, verbose=0, 
        optimization_level=1, set_attributes=False):
        """ Evaluates misfit on given data
        """

        if optimization_level==0 or set_attributes:
            return simple.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_max, verbose, set_attributes)

        if optimization_level==1:
            return fast1.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_max, verbose)

        if optimization_level==2:
            return fast2.misfit(
                data, greens, sources, self.norm, self.time_shift_groups,
                self.time_shift_max, verbose)



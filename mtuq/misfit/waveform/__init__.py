
import numpy as np

from copy import deepcopy
from mtuq.misfit.waveform import level0, level1, level2
from mtuq.misfit.waveform._stats import estimate_sigma, calculate_norm_data
from mtuq.util import Null, iterable, warn
from mtuq.util.math import isclose, list_intersect_with_indices
from mtuq.util.signal import check_padding, get_components, isempty


class Misfit(object):
    """ Waveform misfit function

    Evaluates misfit between data and synthetics using time shifts followed by
    waveform differences. This approach, due to `ZhaoHelmberger1994` and 
    `ZhuHelmberger1996`, has become widely used in regional seismology.  See 
    `docs/` and `docs/references` for more information.

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = Misfit(**parameters)
        values = function_handle(data, greens, sources)

    In the first step, the user supplies parameters such as the type of norm
    (see below for detailed argument descriptions).

    In the second step, the user supplies data, Green's functions, and sources.
    Synthetics are then generated and compared with data, and a NumPy array of 
    misfit values is returned of the same length as `sources`.


    .. rubric:: Parameters

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

    ``optimization_level`` (`int`): optimization level 
    (see further details below)


    .. note:: 

      *Convention* : A positive time shift means synthetics are arriving too 
      early and need to be shifted forward in time to match the observed data.


    .. rubric:: Optimization Levels

    Because misfit evaluation is our most computationally expensive task, we 
    have implemented three different versions: 

    - a readable pure Python version (``mtuq.misfit.level0``)

    - a fast pure Python version (``mtuq.misfit.level1``)

    - a very fast Python/C version (``mtuq.misfit.level2``)


    While having exactly the same input argument syntax, these three versions
    differ in the following ways:

    - ``level0`` provides a reference for understanding what the code is doing
      and for checking the correctness of the fast implementations

    - ``level1`` is an optimized pure Python implementation which provides 
      significant computational savings for `len(sources)` > 100. This
      version is the closest to `Zhu1996`'s original C software.

    - ``level2`` is an optimized Python/C implementation, in which a Python 
      wrapper is used to combine ObsPy traces into multidimensional arrays.
      These arrays are passed to a C extension module, which does the
      main computational work. Unlike the other two versions, this 
      implementation requires that all ObsPy traces have the same time
      discretization.


    .. note:: 

      During installation, C extension modules are automatically compiled by
      `build_ext.sh` using compiler flags given in `setup.py`.  For performance
      tuning or compiler troubleshooting, users may wish to modify the
      `get_compiler_args` function in `setup.py`.

    """

    def __init__(self,
        norm='hybrid',
        time_shift_groups=['ZRT'],
        time_shift_min=0.,
        time_shift_max=0.,
        optimization_level=2,
        ):
        """ Function handle constructor
        """

        if norm.lower()=='hybrid':
            norm = 'hybrid'

        assert norm in ['L1', 'L2', 'hybrid'],\
            ValueError("Bad input argument: norm")

        assert time_shift_max >= 0.,\
            ValueError("Bad input argument: time_shift_max")

        if norm=='L1':
            warn(
                "Consider using norm='hybrid', which is much faster than "
                "norm='L1' but still robust against outliers."
                )

        if type(time_shift_groups) not in (list, tuple):
            raise TypeError

        for group in time_shift_groups:
            for component in group:
                assert component in ['Z','R','T'],\
                    ValueError("Bad input argument")

        assert optimization_level in [0,1,2]

        self.norm = norm
        self.time_shift_min = time_shift_min
        self.time_shift_max = time_shift_max
        self.time_shift_groups = time_shift_groups
        self.optimization_level = optimization_level


    def __call__(self, data, greens, sources, progress_handle=Null(), 
        set_attributes=False, optimization_level=None):
        """ Evaluates misfit on given data
        """
        if optimization_level is None:
            optimization_level = self.optimization_level

        assert optimization_level in [0,1,2]

        # normally misfit is evaluated over a grid of sources; `iterable`
        # makes things work if just a single source is given
        sources = iterable(sources)

        # checks that dataset is nonempty
        if isempty(data):
            warn("Empty data set. No misfit evaluations will be carried out")
            return np.zeros((len(sources), 1))

        # checks that the container legnths are consistent
        if len(data) != len(greens):
            raise Exception("Inconsistent container lengths\n\n  "+
                "len(data): %d\n  len(greens): %d\n" %
                (len(data), len(greens)))
 

        # checks that optional Green's function padding is consistent with time 
        # shift bounds
        check_padding(greens, self.time_shift_min, self.time_shift_max)

 
        if optimization_level==0 or set_attributes:
            return level0.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_min, self.time_shift_max, progress_handle,
                set_attributes)

        if optimization_level==1:
            return level1.misfit(
                data, greens, sources, self.norm, self.time_shift_groups, 
                self.time_shift_min, self.time_shift_max, progress_handle)

        if optimization_level==2:
            return level2.misfit(
                data, greens, sources, self.norm, self.time_shift_groups,
                self.time_shift_min, self.time_shift_max, progress_handle)


    def collect_attributes(self, data, greens, source):
        """ Collects misfit, time shifts and other attributes corresponding to 
        each trace
        """
        # checks that dataset is nonempty
        if isempty(data):
            warn("Empty data set. No attributes will be returned")
            return []

        # checks that optional Green's function padding is consistent with time 
        # shift bounds
        check_padding(greens, self.time_shift_min, self.time_shift_max)

        synthetics = greens.get_synthetics(
            source, components=data.get_components(), mode='map', inplace=True)

        # attaches attributes to synthetics
        _ = level0.misfit(
            data, greens, iterable(source), self.norm, self.time_shift_groups,
            self.time_shift_min, self.time_shift_max, msg_handle=Null(),
            set_attributes=True)

        # collects attributes
        attrs = []
        for stream in synthetics:
            attrs += [{}]
            for trace in stream:
                component = trace.stats.channel[-1]
                if component in attrs[-1]:
                    print('Warning multiple traces for same component')
                if hasattr(trace, 'attrs'):
                    attrs[-1][component] = trace.attrs

        return deepcopy(attrs)


    def collect_synthetics(self, data, greens, source):
        """ Collects synthetics with misfit, time shifts and other attributes attached
        """
        # checks that dataset is nonempty
        if isempty(data):
            warn("Empty data set. No attributes will be returned")
            return []

        # checks that optional Green's function padding is consistent with time 
        # shift bounds
        check_padding(greens, self.time_shift_min, self.time_shift_max)

        synthetics = greens.get_synthetics(
            source, components=data.get_components(), mode='map', inplace=True)

        # attaches attributes to synthetics
        _ = level0.misfit(
            data, greens, iterable(source), self.norm, self.time_shift_groups,
            self.time_shift_min, self.time_shift_max, msg_handle=Null(),
            set_attributes=True)

        return deepcopy(synthetics)



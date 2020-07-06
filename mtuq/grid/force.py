

import numpy as np

from numpy import pi as PI
from numpy.random import uniform as random

from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import asarray
from mtuq.util.math import open_interval as regular
from mtuq.util.lune import to_force


def ForceGridRegular(magnitudes_in_N=1., npts_per_axis=80):
    """ Force grid with regularly-spaced values

    Given input parameters ``magnitudes_in_N`` (`list`) and
    ``npts_per_axis`` (`int`), returns a ``Grid`` of size
    `len(magnitudes_in_N)*npts_per_axis^2`.

    .. rubric :: Usage

    Use ``get(i)`` to return the i-th force as a vector
    `Frr, Ftt, Fpp`

    Use ``get_dict(i)`` to return the i-th force as dictionary
    of parameters `F0, theta, h` (magnitude, azimuth, cos(colatitude)).

    """
    theta = regular(0., 360., npts_per_axis)
    h = regular(-1., 1., npts_per_axis)
    F0 = asarray(magnitudes_in_N)

    return Grid(
        dims=('F0', 'theta', 'h'),
        coords=(F0, theta, h),
        callback=to_force)


def ForceGridRandom(magnitudes_in_N=1., npts=10000):
    """ Force grid with randomly-spaced values

    Given input parameters ``magnitudes_in_N`` (`list`) and
    ``npts`` (`int`), returns an ``UnstructuredGrid`` of size
    `npts*len(magnitudes_in_N)`.

    .. rubric :: Usage

    Use ``get(i)`` to return the i-th force as a vector
    `Frr, Ftt, Fpp`

    Use ``get_dict(i)`` to return the i-th force as dictionary
    of parameters `F0, theta, h` (magnitude, azimuth, cos(colatitude)).

    """
    theta = random(0., 360., npts)
    h = random(-1., 1., npts)
    F0 = asarray(magnitudes_in_N)

    theta = np.tile(theta, len(magnitudes_in_N))
    h = np.tile(h, len(magnitudes_in_N))
    F0 = np.repeat(F0, npts)

    return UnstructuredGrid(
        dims=('F0', 'theta', 'h'),
        coords=(F0, theta, h),
        callback=to_force)

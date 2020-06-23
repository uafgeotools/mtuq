

import numpy as np

from numpy import pi as PI
from numpy.random import uniform as random

from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import asarray
from mtuq.util.math import open_interval as regular
from mtuq.util.lune import to_rtp



def ForceGridRegular(magnitudes_in_N=1., npts_per_axis=80):
    """ Force grid with regularly-spaced values
    """
    theta = regular(0., 360., npts)
    h = regular(-1., 1., npts)
    F0 = asarray(magnitudes_in_N)

    return Grid(
        dims=('F0', 'theta', 'h'),
        coords=(F0, theta, h),
        callback=to_rtp)


def ForceGridRandom(magnitudes_in_N=1., npts=10000):
    """ Force grid with randomly-spaced values
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
        callback=to_rtp)

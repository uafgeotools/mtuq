

import numpy as np

from numpy import pi as PI
from numpy.random import uniform as random

from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import asarray
from mtuq.util.math import open_interval as regular
from mtuq.util.lune import to_mt, to_rho, semiregular_grid



def FullMomentTensorGridRandom(magnitudes=[1.], npts=1000000):
    """ Full moment tensor grid with randomly-spaced values

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`), 
    returns an ``UnstructuredGrid`` of size `npts*len(magnitudes)`.

    Moment tensors are drawn from the uniform distribution defined by 
    `Tape2015` (https://doi.org/10.1093/gji/ggv262)

    .. rubric :: Usage

    Use ``get(i)`` to return the i-th moment tensor as a vector
    `Mrr, Mtt, Mpp, Mrp, Mrt, Mtp`

    Use ``get_dict(i)`` to return the i-th moment tensor as dictionary
    of Tape2015 parameters `rho, v, w, kappa, sigma, h`
    """

    v = random(-1./3., 1./3., npts)
    w = random(-3./8.*PI, 3./8.*PI, npts)
    kappa = random(0., 360, npts)
    sigma = random(-90., 90., npts)
    h = random(0., 1., npts)
    rho = list(map(to_rho, asarray(magnitudes)))

    v = np.tile(v, len(magnitudes))
    w = np.tile(w, len(magnitudes))
    kappa = np.tile(kappa, len(magnitudes))
    sigma = np.tile(sigma, len(magnitudes))
    h = np.tile(h, len(magnitudes))
    rho = np.repeat(rho, npts)
    
    return UnstructuredGrid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)



def FullMomentTensorGridSemiregular(magnitudes=[1.], npts_per_axis=20, tightness=0.8):
    """ Full moment tensor grid with semiregular values

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`), 
    returns a ``Grid`` of size `2*len(magnitudes)*npts_per_axis^5`.

    For tightness~0, the grid will be regular in Tape2015 parameters `v, w`,
    and for tightness~1, regular in Tape2012 parameters `delta, gamma`.
    For intermediate values, the grid will be "semiregular" in the sense of
    a linear interpolation between the above cases (in such a way that, as
    `tightness` increases, the extremal grid points get closer to the boundary
    of the lune).

    .. rubric :: Usage

    Use ``get(i)`` to return the i-th moment tensor as a vector
    `Mrr, Mtt, Mpp, Mrp, Mrt, Mtp`

    Use ``get_dict(i)`` to return the i-th moment tensor as dictionary
    of Tape2015 parameters `rho, v, w, kappa, sigma, h`

    """
    v, w = semiregular_grid(npts_per_axis, 2*npts_per_axis, tightness)

    kappa = regular(0., 360, npts_per_axis)
    sigma = regular(-90., 90., npts_per_axis)
    h = regular(0., 1., npts_per_axis)
    rho = list(map(to_rho, asarray(magnitudes)))

    return Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)


def DoubleCoupleGridRandom(magnitudes=[1.], npts=50000):
    """ Double-couple moment tensor grid with randomly-spaced values

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`), 
    returns an ``UnstructuredGrid`` of size `npts*len(magnitudes)`.

    .. rubric :: Usage

    Use ``get(i)`` to return the i-th moment tensor as a vector
    `Mrr, Mtt, Mpp, Mrp, Mrt, Mtp`

    Use ``get_dict(i)`` to return the i-th moment tensor as dictionary
    of Tape2015 parameters `rho, v, w, kappa, sigma, h`

    """
    v = np.zeros(npts)
    w = np.zeros(npts)
    kappa = random(0., 360, npts)
    sigma = random(-90., 90., npts)
    h = random(0., 1., npts)
    rho = list(map(to_rho, asarray(magnitudes)))

    v = np.tile(v, len(magnitudes))
    w = np.tile(w, len(magnitudes))
    kappa = np.tile(kappa, len(magnitudes))
    sigma = np.tile(sigma, len(magnitudes))
    h = np.tile(h, len(magnitudes))
    rho = np.repeat(rho, npts)

    return UnstructuredGrid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)


def DoubleCoupleGridRegular(magnitudes=[1.], npts_per_axis=40):
    """ Double-couple moment tensor grid with regularly-spaced values

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`), 
    returns a ``Grid`` of size `len(magnitudes)*npts_per_axis^3`.

    .. rubric :: Usage

    Use ``get(i)`` to return the i-th moment tensor as a vector
    `Mrr, Mtt, Mpp, Mrp, Mrt, Mtp`

    Use ``get_dict(i)`` to return the i-th moment tensor as dictionary
    of Tape2015 parameters `rho, v, w, kappa, sigma, h`
    """ 
    v = 0.
    w = 0.

    kappa = regular(0., 360, npts_per_axis)
    sigma = regular(-90., 90., npts_per_axis)
    h = regular(0., 1., npts_per_axis)
    rho = list(map(to_rho, asarray(magnitudes)))

    return Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)



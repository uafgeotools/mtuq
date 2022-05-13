

import numpy as np

from numpy import pi
from numpy.random import uniform as random

from mtuq.event import Force, MomentTensor
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import asarray
from mtuq.util.math import open_interval as regular
from mtuq.util.math import to_mij, to_rho, semiregular_grid, to_v, to_w


# How to use grids
#
#    Use ``get(i)`` to return the `i`-th moment tensor as a `MomentTensor` object
#
#    Use ``get(i).as_vector()`` to return the `i`-th moment tensor as a vector
#    `Mrr, Mtt, Mpp, Mrp, Mrt, Mtp`
#
#    Use ``get_dict(i)`` to return the `i`-th moment tensor as dictionary
#    of Tape2015 parameters `rho, v, w, kappa, sigma, h`



def FullMomentTensorGridRandom(magnitudes=[1.], npts=1000000):
    """ Grid with randomly-drawn full moment tensors

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`),
    returns an ``UnstructuredGrid`` of size `npts*len(magnitudes)`.

    Moment tensors are drawn from the uniform distribution defined by
    `Tape2015 <https://uafgeotools.github.io/mtuq/references.html>`_

    """

    v = random(-1./3., 1./3., npts)
    w = random(-3./8.*pi, 3./8.*pi, npts)
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


def FullMomentTensorGridSemiregular(magnitudes=[1.], npts_per_axis=20, 
    tightness=0.8, uniformity=0.8):
    """ Grid with regularly-spaced full moment tensors


    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`),
    returns a ``Grid`` of size `2*len(magnitudes)*npts_per_axis^5`.

    The grid is `semiregular` in the sense of an interpolation between
    two parameterizations.  `See here
    <mtuq.grid.moment_tensor.semiregular_grid.html>`_ for details.

    """
    v, w = semiregular_grid(npts_per_axis, 2*npts_per_axis, tightness, uniformity)

    kappa = regular(0., 360, npts_per_axis)
    sigma = regular(-90., 90., npts_per_axis)
    h = regular(0., 1., npts_per_axis)
    rho = list(map(to_rho, asarray(magnitudes)))

    return Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)


def DeviatoricGridRandom(magnitudes=[1.], npts=100000):
    """ Grid with randomly-drawn deviatoric moment tensors

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`),
    returns an ``UnstructuredGrid`` of size `npts*len(magnitudes)`.

    Moment tensors are drawn from the uniform distribution defined by
    `Tape2015 <https://uafgeotools.github.io/mtuq/references.html>`_

    """

    v = random(-1./3., 1./3., npts)
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

    return Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)



def DeviatoricGridSemiregular(magnitudes=[1.], npts_per_axis=20,
    tightness=0.8, uniformity=0.8):
    """ Grid with reguarly-spaced deviatoric moment tensors

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`),
    returns a ``Grid`` of size `len(magnitudes)*npts_per_axis^4`.

    The grid is `semiregular` in the sense of an interpolation between
    two parameterizations.  `See here
    <mtuq.grid.moment_tensor.semiregular_grid.html>`_ for details.

    """

    v, _ = semiregular_grid(npts_per_axis, 1, tightness, uniformity)
    w = 0

    kappa = regular(0., 360, npts_per_axis)
    sigma = regular(-90., 90., npts_per_axis)
    h = regular(0., 1., npts_per_axis)
    rho = list(map(to_rho, asarray(magnitudes)))

    return Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)


def DoubleCoupleGridRandom(magnitudes=[1.], npts=50000):
    """ Grid with randomly-drawn double couple moment tensors

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`),
    returns an ``UnstructuredGrid`` of size `npts*len(magnitudes)`.

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
    """ Grid with reguarly-spaced double couple moment tensors

    Given input parameters ``magnitudes`` (`list`) and ``npts`` (`int`),
    returns a ``Grid`` of size `len(magnitudes)*npts_per_axis^3`.

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


#
# utility functions
#

def to_mt(rho, v, w, kappa, sigma, h):
    """ Converts from lune parameters to MomentTensor object
    """
    mt = to_mij(rho, v, w, kappa, sigma, h)
    return MomentTensor(mt, convention='USE')




#
# deprecated
#

def FullMomentTensorPlottingGrid(magnitudes=[1.], npts_per_axis=11):
    # depracated because separate tightness and uniformity options have been
    # added to FullMomentTensorGridSemiregular

    v1, v2, nv = -30, 30, 13
    w1, w2, nw = -1, 1, 35
    dv = (v2-v1)/nv
    dw = (w2-w1)/nw
    v = np.arange((v1+dv/2), (v2-dv/2), dv)
    w = np.arange((w1+dw/2), (w2+dw/2), dw)
    gamma = v
    delta = np.arcsin(w)*(180/pi)
    v,w = to_v(gamma), to_w(delta)

    kappa = regular(0., 360, npts_per_axis)
    sigma = regular(-90., 90., npts_per_axis)
    h = regular(0., 1., npts_per_axis)
    rho = list(map(to_rho, asarray(magnitudes)))

    return Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho, v, w, kappa, sigma, h),
        callback=to_mt)


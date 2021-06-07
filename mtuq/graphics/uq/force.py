
#
# graphics/uq/force.py - uncertainty quantification of forces on the unit sphere
#

import numpy as np

from matplotlib import pyplot

from mtuq.graphics.uq._gmt import gmt_plot_force
from mtuq.grid_search import DataFrame, DataArray, MTUQDataArray, MTUQDataFrame
from mtuq.util import warn
from mtuq.util.math import closed_interval, open_interval


def plot_misfit_force(filename, ds, **kwargs):
    """ Plots misfit values with respect to force orientation (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    See _plot_force for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'viridis',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = calculate_misfit(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = calculate_misfit_unstruct(ds)

    _plot_force(filename, misfit, **kwargs)


def plot_likelihood_force(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values with respect to force orientation 
    (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    See _plot_force for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        likelihoods = calculate_likelihoods(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = calculate_likelihoods_unstruct(ds, var)

    _plot_force(filename, likelihoods, **kwargs)


def plot_marginal_force(filename, ds, var, **kwargs):
    """ Plots marginal likelihood values with respect to force orientation 
    (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    See _plot_force for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = calculate_marginals(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = calculate_marginals_unstruct(ds, var)

    _plot_force(filename, marginals, **kwargs)


#
# backend
#

def _plot_force(filename, da, show_best=True, show_tradeoffs=False, **kwargs):
    """ Plots values with respect to force orientation (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray`):
    DataArray containing `v,w` values and corresponding misfit values

    """
    if not issubclass(type(da), DataArray):
        raise Exception()


    best_force = None
    if show_best:
        if 'best_force' in da.attrs:
            best_vw = da.attrs['best_force']
        else:
            warn("Best-fitting force not given")

    gmt_plot_force(filename,
        da.coords['phi'],
        da.coords['h'], 
        da.values.transpose(),
        best_force=best_force,
        **kwargs)


#
# for extracting misfit or likelihood from regularly-spaced grids
#

def calculate_misfit(da):
    """ For each point on lune, extracts minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'F0'))

    return misfit#.assign_attrs({
        #})


def calculate_likelihoods(da, var):
    """ For each point on lune, calculates maximum likelihood value
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'F0'))
    likelihoods.values /= 4.*np.pi*likelihoods.values.sum()

    return likelihoods#.assign_attrs({
        #})


def calculate_marginals(da, var):
    """ For each point on lune, calculates marginal likelihood value
    """

    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    marginals = likelihoods.sum(dim=('origin_idx', 'F0'))
    marginals.values /= 4.*np.pi*marginals.values.sum()

    return marginals#.assign_attrs({
        #})


#
# for extracting misfit or likelihood from irregularly-spaced grids
#

def calculate_misfit_unstruct(df, **kwargs):
    df = df.copy()
    df = df.reset_index()
    da = _bin(df, lambda df: df.min(), **kwargs)

    return da#.assign_attrs({
        #})


def calculate_likelihoods_unstruct(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = _bin(df, lambda df: df.max(), **kwargs)
    da.values /= 4.*np.pi*da.values.sum()

    return da#.assign_attrs({
        #})


def calculate_marginals_unstruct(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = _bin(df, lambda df: df.sum()/len(df))
    da.values /= 4.*np.pi*da.values.sum()

    return da#.assign_attrs({
        #})


#
# bins irregularly-spaced forces into phi,h rectangles
#

def _bin(df, handle, npts_phi=60, npts_h=30):
    """ Bins DataFrame into rectangular cells
    """
    # define centers of cells
    centers_phi = open_interval(0., 360., npts_phi)
    centers_h = open_interval(-1., +1., npts_h)

    # define corners of cells
    phi = closed_interval(0., 360, npts_phi+1)
    h = closed_interval(-1., +1., npts_h+1)

    binned = np.empty((npts_h, npts_phi))
    for _i in range(npts_h):
        for _j in range(npts_phi):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['phi'].between(phi[_j], phi[_j+1]) &
                df['h'].between(h[_i], h[_i+1])]

            if len(subset)==0:
                print("Encountered empty bin\n"
                      "phi: %f, %f\n"
                      "h: %f, %f\n" %
                      (phi[_j], phi[_j+1], h[_i], h[_i+1]) )

            binned[_i, _j] = handle(subset[0])

    return DataArray(
        dims=('phi', 'h'),
        coords=(centers_phi, centers_h),
        data=binned.transpose()
        )



#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


def _defaults(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
           kwargs[key] = defaults[key]


#
# For details about the eigenvalue lune, see
# Tape2012 - A geometric setting for moment tensors
# (https://doi.org/10.1111/j.1365-246X.2012.05491.x)
#

import numpy as np

from matplotlib import pyplot

from pandas import DataFrame
from xarray import DataArray
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import dataarray_idxmin, dataarray_idxmax

from mtuq.graphics.uq._gmt import exists_gmt, gmt_not_found_warning, \
    gmt_plot_misfit_lune, gmt_plot_likelihood_lune, gmt_plot_misfit_mt_lune
from mtuq.graphics.uq.vw import _bin
from mtuq.util import warn
from mtuq.util.math import apply_cov, lune_det, to_gamma, to_delta, to_v, to_w, \
    semiregular_grid, to_mij, to_Mw

gmt_plot_lune = gmt_plot_misfit_lune


def plot_misfit_lune(filename, ds, **kwargs):
    """ Plots misfit values on eigenvalue lune (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    See _plot_lune for keyword argument descriptions

    """ 
    _defaults(kwargs, {
        'colormap': 'viridis',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = extract_misfit(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = extract_misfit_unstruct(ds)        

    _plot_lune(filename, misfit, **kwargs)


def plot_likelihood_lune(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values on eigenvalue lune (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    See _plot_lune for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'hot',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        likelihoods = calculate_likelihoods(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = calculate_likelihoods_unstruct(ds, var)

    _plot_lune(filename, likelihoods, **kwargs)


def plot_marginal_lune(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values on eigenvalue lune (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    See _plot_lune for keyword argument descriptions
    """
    _defaults(kwargs, {
        'colormap': 'hot',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = calculate_marginals(ds, var)

    elif issubclass(type(ds), DataFrame):
        warn("plot_marginal_lune not implemented for irregular grids")
        #marginals = calculate_marginals_unstruct(ds, var)

    _plot_lune(filename, marginals, **kwargs)


#
# a more general plotting function
#

def _plot_lune(filename, da, show_best=True, show_tradeoffs=False, **kwargs):
    """ Plots DatArray values on the eigenvalue lune (requires GMT)

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray`):
    DataArray containing `v,w` values and corresponding misfit values

    """
    if not issubclass(type(da), DataArray):
        raise Exception()


    best_vw = None
    mt_array = None

    if show_best:
        if 'best_vw' in da.attrs:
            best_vw = da.attrs['best_vw']
        else:
            warn("Best-fitting moment tensor not given")

    if show_tradeoffs:
        if 'mt_array' in da.attrs:
            mt_array = da.attrs['mt_array']
        else:
            warn("Tradeoffs not given")


    gmt_plot_lune(filename, 
        to_gamma(da.coords['v']), 
        to_delta(da.coords['w']),
        da.values.transpose(), 
        best_vw=best_vw,
        mt_array=mt_array,
        **kwargs)


#
# for extracting misfit or likelihood from regularly-spaced grids
#

def extract_misfit(da):
    """ For each point on lune, extracts minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))

    return misfit.assign_attrs({
        'best_mt': _best_mt(da),
        'best_vw': _best_vw(da),
        'mt_array': extract_mt_array(da),
        })


def calculate_likelihoods(da, var):
    """ For each point on lune, calculates maximum likelihood value
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods = likelihoods.max(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))

    return likelihoods.assign_attrs({
        'best_mt': _best_mt(da),
        'best_vw': _best_vw(da),
        'mt_array': extract_mt_array(da),
        'likelihood_max': likelihoods.max(),
        'likelihood_vw': dataarray_idxmax(likelihoods).values(),
        })


def calculate_marginals(ds, sigma):
    """ For each point on lune, calculates marginal likelihood value
    """
    raise NotImplementedError


#
# for extracting misfit or likelihood from irregularly-spaced grids
#

def extract_misfit_unstruct(df, **kwargs):
    df = df.copy()
    df = df.reset_index()
    return _bin(df, lambda df: df.min(), **kwargs)


def calculate_likelihoods_unstruct(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()
    return _bin(df, lambda df: df.max(), **kwargs)


def calculate_marginals_unstruct(df, var, **kwargs):
    raise NotImplementedError


#
# examples of showing how other quantities, besides misfit or likelihood, can
# be extracted from regularly-spaced grids
#

def extract_magnitudes(da):
    """ For each point on lune, extracts magnitude of best-fitting moment tensor
    """
    #
    # TODO - generalize for multiple orgins
    #
    origin_idx = 0

    nv = len(ds.coords['v'])
    nw = len(ds.coords['w'])

    rho_array = np.empty((nv,nw))
    for iv in range(nv):
        for iw in range(nw):
            sliced = da[:,iv,iw,:,:,:,origin_idx]
            argmin = np.argmin(sliced.values, axis=None)
            idx = np.unravel_index(argmin, np.shape(sliced))

            rho_array[iv,iw] = da['rho'][idx[0]]

    return to_Mw(rho)


def extract_mt_array(da):
    """ For each point on lune, collects best-fitting moment tensor
    """
    #
    # TODO - generalize for multiple orgins
    #
    origin_idx = 0

    nv = len(da.coords['v'])
    nw = len(da.coords['w'])

    mt_array = np.empty((nv,nw,6))
    for iv in range(nv):
        for iw in range(nw):
            sliced = da[:,iv,iw,:,:,:,origin_idx]
            argmin = np.argmin(sliced.values, axis=None)
            idx = np.unravel_index(argmin, np.shape(sliced))

            mt_array[iv,iw,0] = da['rho'][idx[0]]
            mt_array[iv,iw,1] = da['v'][iv]
            mt_array[iv,iw,2] = da['w'][iw]
            mt_array[iv,iw,3] = da['kappa'][idx[1]]
            mt_array[iv,iw,4] = da['sigma'][idx[2]]
            mt_array[iv,iw,5] = da['h'][idx[3]]

    return mt_array


def calculate_variance_reduction(da, reference_value):
    """ For each point on lune, calculates variance reduction
    """
    misfit = da.min(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
    return (reference_value - misfit)/reference_value


def _best_mt(da):
    """ Returns overall best-fitting moment tensor
    """
    da = dataarray_idxmin(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _best_vw(da):
    """ Returns overall best v,w
    """
    da = dataarray_idxmin(da)
    lune_keys = ['v', 'w']
    lune_vals = [da[key].values for key in lune_keys]
    return lune_vals


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



#
# For details about the eigenvalue lune, see
# Tape2012 - A geometric setting for moment tensors
# (https://doi.org/10.1111/j.1365-246X.2012.05491.x)
#

import numpy as np
import pandas
import xarray

from matplotlib import pyplot
from mtuq.grid_search import DataArray, DataFrame, MTUQDataArray, MTUQDataFrame
from mtuq.graphics.uq._gmt import gmt_plot_lune
from mtuq.util import warn
from mtuq.util.math import lune_det, to_gamma, to_delta

from mtuq.graphics.uq.vw import\
    calculate_misfit, calculate_misfit_unstruct,\
    calculate_likelihoods, calculate_likelihoods_unstruct,\
    calculate_marginals, calculate_marginals_unstruct


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
        misfit = calculate_misfit(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = calculate_misfit_unstruct(ds)        

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
        'colormap': 'hot_r',
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
        'colormap': 'hot_r',
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = calculate_marginals(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = calculate_marginals_unstruct(ds, var)

    _plot_lune(filename, marginals, **kwargs)



#
# backend
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
    lune_array = None

    if show_best:
        if 'best_vw' in da.attrs:
            best_vw = da.attrs['best_vw']
        else:
            warn("Best-fitting moment tensor not given")

    if show_tradeoffs:
        if 'lune_array' in da.attrs:
            lune_array = da.attrs['lune_array']
        else:
            warn("Focal mechanism tradeoffs not given")


    gmt_plot_lune(filename, 
        to_gamma(da.coords['v']), 
        to_delta(da.coords['w']),
        da.values.transpose(), 
        best_vw=best_vw,
        lune_array=lune_array,
        **kwargs)


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



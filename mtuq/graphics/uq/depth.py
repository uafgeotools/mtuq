
#
# graphics/uq/origin.py - uncertainty quantification of source origin
#

import numpy as np
import subprocess

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics.uq._gmt import gmt_plot_depth
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import fullpath, warn
from mtuq.util.math import closed_interval, open_interval



def plot_misfit_depth(filename, ds, origins, **kwargs):
    """ Plots misfit versus depth

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``origins`` (`list` of `Origin` objects)
    Origin objects corresponding to different depths


    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_depth.html>`_

    """
    _defaults(kwargs, {
        'ylabel': 'Misfit',
        })


    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        da = _misfit_regular(ds)

    elif issubclass(type(ds), DataFrame):
        da = _misfit_random(ds)

    _plot_depth(filename, da, origins, **kwargs)



def plot_likelihood_depth(filename, ds, origins, var=None, **kwargs):
    """ Plots maximum likelihoods versus depth

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``origins`` (`list` of `Origin` objects)
    Origin objects corresponding to different depths


    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_depth.html>`_

    """

    raise NotImplementedError



def plot_marginal_depth(filename, ds, origins, var=None, **kwargs):
    """ Plots marginal likelihoods versus depth

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``origins`` (`list` of `Origin` objects)
    Origin objects corresponding to different depths


    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_depth.html>`_

    """

    raise NotImplementedError



#
# for extracting values from regularly-spaced grids
#

def _misfit_regular(da):
    dims = ('rho', 'v', 'w', 'kappa', 'sigma', 'h')
    return da[da.argmin(dims)]


def _likelihoods_regular(da, var):
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    dims = ('rho', 'v', 'w', 'kappa', 'sigma', 'h')
    idx = likelihoods.argmax(dims)
    return likelihoods[idx]


#
# wrappers
#

def _backend(filename,
        depths,
        values,
        magnitudes=None,
        lune_array=None,
        title=None,
        xlabel=None,
        ylabel=None,
        fontsize=16.):

    figsize = (6., 6.)
    pyplot.figure(figsize=figsize)
    pyplot.plot(depths, values, 'k-')

    if title:
        pyplot.title(title, fontsize=fontsize)

    if xlabel:
         pyplot.xlabel(xlabel, fontsize=fontsize)

    if ylabel:
         pyplot.ylabel(ylabel, fontsize=fontsize)

    pyplot.savefig(filename)


def _plot_depth(filename, da, origins, title='',
    xlabel='auto', ylabel='', show_magnitudes=False, show_tradeoffs=False,
    backend=gmt_plot_depth):

    """ Plots depth versus user-supplied DataArray values (requires GMT)

    .. rubric :: Keyword arguments

    ``show_magnitudes`` (`bool`):
    Write magnitude annotation for each plotted value

    ``show_tradeoffs`` (`bool`):
    Show how focal mechanism trades off with depth

    ``xlabel`` (`str`):
    Optional x-axis label

    ``ylabel`` (`str`):
    Optional y-axis label

    ``title`` (`str`)
    Optional figure title

    """

    npts = len(origins)

    depths = np.empty(npts)
    values = np.empty(npts)
    for _i, origin in enumerate(origins):
        depths[_i] = origin.depth_in_m
        values[_i] = da.values[_i]

    magnitudes = None
    if show_magnitudes:
        magnitudes = np.empty(npts)
        for _i in range(npts):
            magnitudes[_i] = da[_i].coords['rho']

    lune_array = None
    if show_tradeoffs:
        lune_array = np.empty((npts, 6))
        for _i in range(npts):
            lune_array[_i, 0] = da[_i].coords['rho']
            lune_array[_i, 1] = da[_i].coords['v']
            lune_array[_i, 2] = da[_i].coords['w']
            lune_array[_i, 3] = da[_i].coords['kappa']
            lune_array[_i, 4] = da[_i].coords['sigma']
            lune_array[_i, 5] = da[_i].coords['h']

    if xlabel=='auto' and (depths.max() < 10000.):
       xlabel = 'Depth (m)'
    elif xlabel=='auto' and (depths.max() >= 10000.):
       depths /= 1000.
       xlabel = 'Depth (km)'

    backend(filename,
        depths,
        values,
        magnitudes=magnitudes,
        lune_array=lune_array,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
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



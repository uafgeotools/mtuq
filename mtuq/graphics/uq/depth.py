
#
# graphics/uq/origin.py - uncertainty quantification of source origin
#

import numpy as np
import subprocess

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics.uq._gmt import exists_gmt, gmt_not_found_warning
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import fullpath, warn
from mtuq.util.math import closed_interval, open_interval



def plot_misfit_depth(filename, ds, origins, title='', **kwargs):
    """ Plots misfit versus depth

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values


    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_depth.html>`_

    """
    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        da = _misfit_regular(ds)

    elif issubclass(type(ds), DataFrame):
        da = _misfit_random(ds)

    _plot_depth(filename, origins, da, title, **kwargs)



def plot_likelihood_depth(filename, ds, origins, sources, sigma=None, title=''):
    """ Plots maximum likelihood versus depth

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title

    """
    assert sigma is not None

    depths = _get_depths(origins)

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        ds.values = np.exp(-ds.values/(2.*sigma**2))
        ds.values /= ds.values.sum()

        values, indices = _min_dataarray(ds)
        best_sources = _get_sources(sources, indices)

    elif issubclass(type(ds), DataFrame):
        ds = np.exp(-ds/(2.*sigma**2))
        ds /= ds.sum()

        values, indices = _min_dataframe(ds)
        best_sources = _get_sources(sources, indices)

    values /= values.sum()

    _plot_depth(filename, depths, values, indices, 
        title=title, xlabel='auto', ylabel='Likelihood')



def plot_marginal_depth(filename, ds, origins, sources, sigma=None, title=''):
    """ Plots marginal likelihoods versus depth


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title

    """
    assert sigma is not None

    depths = _get_depths(origins)

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        ds = np.exp(-ds/(2.*sigma**2))
        ds /= ds.sum()

        values, indices = _max_dataarray(ds)
        best_sources = _get_sources(sources, indices)

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError
        ds = np.exp(-ds/(2.*sigma**2))
        ds /= ds.sum()

        values, indices = _min_dataframe(ds)
        best_sources = _get_sources(sources, indices)

    values /= values.sum()

    _plot_depth(filename, depths, values, indices, 
        title=title, xlabel='auto', ylabel='Likelihood')



#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


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
# pyplot wrappers
#

def _backend(filename,
        depths,
        values,
        magnitudes=None,
        mt_array=None,
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


def _plot_depth(filename, origins, da, title='',
    xlabel='auto', ylabel='', show_magnitudes=True, show_tradeoffs=True,
    fontsize=16., backend=_backend):

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

    mt_array = None
    if show_tradeoffs:
        mt_array = np.empty((npts, 6))
        for _i in range(npts):
            mt_array[_i, 0] = da[_i].coords['rho']
            mt_array[_i, 1] = da[_i].coords['v']
            mt_array[_i, 2] = da[_i].coords['w']
            mt_array[_i, 3] = da[_i].coords['kappa']
            mt_array[_i, 4] = da[_i].coords['sigma']
            mt_array[_i, 5] = da[_i].coords['h']

    if xlabel=='auto' and (depths.max() < 10000.):
       xlabel = 'Depth (m)'
    elif xlabel=='auto' and (depths.max() >= 10000.):
       depths /= 1000.
       xlabel = 'Depth (km)'

    backend(filename,
        depths,
        values,
        magnitudes=magnitudes,
        mt_array=mt_array,
        xlabel=xlabel,
        ylabel=ylabel,
        fontsize=fontsize)



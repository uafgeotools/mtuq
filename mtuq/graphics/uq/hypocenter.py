
#
# graphics/uq/hypocenter.py
#

import numpy as np
import subprocess

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics.uq._gmt import _plot_latlon_gmt
from mtuq.graphics.uq.depth import _misfit_regular, _likelihoods_regular
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import fullpath, warn
from mtuq.util.math import closed_interval, open_interval


def plot_misfit_latlon(filename, ds, origins, **kwargs):
    """ Plots misfit versus hypocenter location

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``origins`` (`list` of `Origin` objects)
    Origin objects corresponding to different hypocenters


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_latlon.html>`_

    """

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        da = _misfit_regular(ds)

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError

    _plot_latlon(filename, da, origins, **kwargs)


def plot_likelihood_latlon(filename, ds, origins, **kwargs):
    """ Plots likelihood versus hypocenter location

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``origins`` (`list` of `Origin` objects)
    Origin objects corresponding to different hypocenters


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_latlon.html>`_

    """

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        da = _likelihood_regular(ds)

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError

    _plot_latlon(filename, da, origins, **kwargs)



def plot_marginal_latlon(filename, ds, origins, **kwargs):
    """ Plots likelihood versus hypocenter location

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``origins`` (`list` of `Origin` objects)
    Origin objects corresponding to different hypocenters


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_depth.html>`_

    """

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        raise NotImplementedError

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError

    _plot_latlon(filename, da, origins, **kwargs)




#
# wrappers
#

def _plot_latlon(filename, da, origins,show_best=False, show_tradeoffs=False,
    backend=_plot_latlon_gmt, **kwargs):

    """ Plots user-supplied DataArray values versus hypocenter (requires GMT)

    .. rubric :: Keyword arguments

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

    lon = np.empty(npts)
    lat = np.empty(npts)
    values = np.empty(npts)

    for _i, origin in enumerate(origins):
        lon[_i] = origin.longitude
        lat[_i] = origin.latitude
        values[_i] = da.values[_i]

    best_latlon = None
    if show_best:
        raise NotImplementedError

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


    backend(filename,
        lon, lat, values,
        best_latlon=best_latlon,
        lune_array=lune_array,
        **kwargs)


def _get_labeltype(x,y,labeltype):
    if labeltype=='latlon':
       xlabel = None
       ylabel = None

    if labeltype=='offset' and ((x.max()-x.min()) >= 10000.):
       x /= 1000.
       y /= 1000.
       xlabel = 'E-W offset (km)'
       ylabel = 'N-S offset (km)'
    elif labeltype=='offset' and ((x.max()-x.min()) < 10000.):
       xlabel = 'E-W offset (m)'
       ylabel = 'N-S offset (m)'

    return xlabel,ylabel


#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


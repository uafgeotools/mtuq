
#
# graphics/uq/hypocenter.py
#

import numpy as np
import subprocess

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics.uq._gmt import gmt_plot_latlon
from mtuq.graphics.uq.depth import _misfit_regular, _likelihoods_regular
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import fullpath, warn
from mtuq.util.math import closed_interval, open_interval


def plot_misfit_latlon(filename, ds, origins, **kwargs):
    """ Plots misfit versus hypocenter position

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title

    """
    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        da = _misfit_regular(ds)

    elif issubclass(type(ds), DataFrame):
        raise NotImplementedError

    _plot_latlon(filename, da, origins)


#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


#
# wrappers
#

def _plot_latlon(filename, da, origins, title='',
    show_best=False, show_tradeoffs=False, **kwargs):

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


    gmt_plot_latlon(filename,
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


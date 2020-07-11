
#
# graphics/uq_vw.py - uncertainty quantification on the v-w rectangle
#

import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util.math import closed_interval, open_interval, to_delta, to_gamma


#
# For details about the  rectangle:`v-w` rectangle rectangle, see 
# Tape2015 - A uniform parameterization of moment tensors
# (https://doi.org/10.1093/gji/ggv262)
#

v_min = -1./3.
v_max = +1./3.
w_min = -3.*np.pi/8.
w_max = +3.*np.pi/8.
vw_area = (v_max-v_min)*(w_max-w_min)



def plot_misfit_vw(filename, ds, title=None):
    """ Plots misfit values on `v-w` rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)

    """
    _check(ds)
    ds = ds.copy()


    if issubclass(type(ds), DataArray):
        da = ds.min(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        v = da.coords['v']
        w = da.coords['w']
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        v, w, values = _bin(df, lambda df: df.min())


    _plot_vw(filename, v, w, values, cmap='hot')



def plot_likelihood_vw(filename, ds, sigma=1., title=None):
    """ Plots maximum likelihoods on `v-w` rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)

    """
    _check(ds)
    ds = ds.copy()


    # convert from misfit to likelihood
    ds.values = np.exp(-ds.values/(2.*sigma**2))
    ds.values /= ds.values.sum()



    if issubclass(type(ds), DataArray):
        da = ds.max(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        v = da.coords['v']
        w = da.coords['w']
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        df['values'] /= df['values'].sum()
        v, w, values = _bin(df, lambda df: df.max())


    values /= values.sum()
    values /= vw_area

    _plot_vw(filename, v, w, values, cmap='hot_r')



def plot_marginal_vw(filename, ds, sigma=1., title=None):
    """ Plots marginal likelihoods on `v-w` rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)


    """
    _check(ds)
    ds = ds.copy()


    # convert from misfit to likelihood
    ds.values = np.exp(-ds.values/(2.*sigma**2))
    ds.values /= ds.values.sum()


    if issubclass(type(ds), DataArray):
        da = ds.sum(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        v = da.coords['v']
        w = da.coords['w']
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        v, w, values = _bin(df, lambda df: df.sum()/len(df))


    values /= values.sum()
    values /= vw_area

    _plot_vw(filename, v, w, values, cmap='hot_r')



def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")



#
# utilities for irregularly-spaced grids
#


def _bin(df, handle, npts_v=20, npts_w=40):
    """ Bins DataFrame into rectangular cells
    """
    # define centers of cells
    centers_v = open_interval(-1./3., 1./3., npts_v)
    centers_w = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    # define corners of cells
    v = closed_interval(-1./3., 1./3., npts_v+1)
    w = closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w+1)

    binned = np.empty((npts_w, npts_v))
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(v[_j], v[_j+1]) &
                df['w'].between(w[_i], w[_i+1])]

            binned[_i, _j] = handle(subset[0])

    return centers_v, centers_w, binned



#
# pyplot wrappers
#

def _plot_vw(filename, v, w, values, cmap='hot'):
    """ Creates `v-w` color plot 

    (Thinly wraps pyplot.pcolor)

    """ 
    fig, ax = pyplot.subplots(figsize=(3., 8.), constrained_layout=True)

    # pcolor requires corners of pixels
    corners_v = _centers_to_edges(v)
    corners_w = _centers_to_edges(w)

    # `values` gets mapped to pixel colors
    pyplot.pcolor(corners_v, corners_w, values, cmap=cmap)

    # v and w have the following bounds
    # (see https://doi.org/10.1093/gji/ggv262)
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.xticks([], [])
    pyplot.yticks([], [])

    pyplot.colorbar(
        orientation='horizontal', 
        ticks=[], 
        pad=0.,
        )

    pyplot.savefig(filename)


def _centers_to_edges(v):

    if issubclass(type(v), DataArray):
        v = v.values.copy()
    else:
        v = v.copy()

    dv = (v[1]-v[0])
    v -= dv/2
    v = np.pad(v, (0, 1))
    v[-1] = v[-2] + dv

    return v



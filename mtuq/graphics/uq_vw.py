
#
# graphics/uq_vw.py - uncertainty quantification on the v-w rectangle
#

#
# For details about the v-w rectangle, see 
# Tape2015 - A uniform parameterization of moment tensors
# (https://doi.org/10.1093/gji/ggv262)
#

#
# These utilities expect  xarray.DataArray format if grid spacing is regular 
# and pandas.DataFrame format if grid spacing is random.  We use these 
# structures because they make data manipulation much easier.  To convert to
# these formats, see
#     mtuq.grid.Grid.to_dataarray
#     mtuq.grid.UnstructuredGrid.to_dataframe
#




import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.math import closed_interval, open_interval
from mtuq.util.xarray import dataarray_to_table



def plot_misfit_vw(filename, struct):
    """ Plots misfit values on 'v-w' rectangle
    """
    _check(struct)
    struct = struct.copy()


    if type(struct)==DataArray:
        da = structure.copy()
        da = da.min(dim=('rho', 'kappa', 'sigma', 'h'))
        v = da.coords['v']
        w = da.coords['w']
        values = da.values


    elif type(struct)==DataFrame:
        df = structure.copy()
        v, w, values = _bin(df, lambda df: df.min())


    _plot_vw(v, w, values, cmap='hot')
    pyplot.savefig(filename)



def plot_likelihood_vw(filename, struct, sigma=1.):
    """ Plots maximum likelihood values on 'v-w' rectangle
    """
    _check(struct)
    struct = struct.copy()


    # convert from misfit to likelihood
    struct.values = np.exp(-struct.values/(2.*sigma**2))


    if type(struct)==DataArray:
        da = struct.max(dim=('rho', 'kappa', 'sigma', 'h'))
        v = da.coords['v']
        w = da.coords['w']
        values = da.values/da.values.sum()


    elif type(struct)==DataFrame:
        df = struct
        df['values'] /= df['values'].sum()
        v, w, values = _bin(df, lambda df: df.max())


    _plot_vw(v, w, values, cmap='hot_r')
    pyplot.savefig(filename)



def plot_marginal_vw(filename, struct, sigma=1.):
    """ Plots marginal likelihood values on 'v-w' rectangle
    """
    _check(struct)
    struct = struct.copy()


    # convert from misfit to likelihood
    struct.values = np.exp(-struct.values/(2.*sigma**2))


    if type(struct)==DataArray:
        da = struct.sum(dim=('rho', 'kappa', 'sigma', 'h'))
        v = da.coords['v']
        w = da.coords['w']
        area = (np.pi/2.)*da.values.sum()
        values = da.values/area


    elif type(struct)==DataFrame:
        df = struct
        v, w, values = _bin(df, lambda df: df.sum()/len(df))
        area = np.pi/2.
        values /= area*df.values.sum()


    _plot_vw(v, w, values, cmap='hot_r')
    pyplot.savefig(filename)



def _check(struct):
    """ Checks data structures
    """
    if type(struct) in (DataArray, DataFrame):
        pass

    elif type(struct) in (Grid, UnstructuredGrid):
        raise TypeError(
            "Plotting utilities expect grid and misfit values in DataArray or "
            "DataFrame format.  For converting to these formats, see "
            "  mtuq.grid.Grid.to_dataarray" 
            "  mutq.grid.UnstructuredGrid.to_dataframe")

    else:
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

            binned[_i, _j] = handle(subset['values'])

    return centers_v, centers_w, binned



#
# pyplot wrappers
#

def _plot_vw(v, w, values, cmap='hot'):
    """ Creates v-w color plot 

    (Thinly wraps pyplot.pcolor)

    """ 
    fig, ax = pyplot.subplots(figsize=(3., 8.), constrained_layout=True)

    nv, nw = len(v), len(w)
    if values.shape == (nv, nw):
        values = values.T

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


def _centers_to_edges(v):
    v = v.copy()
    dv = (v[1]-v[0])
    v -= dv/2
    v = np.pad(v, (0, 1))
    v[-1] = v[-2] + dv
    return v




#
# graphics/uq_vw.py - uncertainty quantification on the v-w rectangle
#

import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.math import closed_interval, open_interval
from mtuq.util.xarray import dataarray_to_table


def plot_misfit_vw(filename, grid, values):
    """ Plots misfit on 'v-w' rectangle
    (matplotlib implementation)
    """
    gridtype = type(grid)

    if gridtype==Grid:
        # convert from mtuq object to xarray DataArray
        da = grid.to_dataarray(values)

        _plot_misfit_regular(filename, da)

    elif gridtype==UnstructuredGrid:
        # convert from mtuq object to pandas Dataframe
        df = grid.to_dataframe(values)

        _plot_misfit_random(filename, df)


def plot_likelihood_vw(filename, grid, values=None):
    """ Plots probability density function on 'v-w' rectangle
    (matplotlib implementation)
    """
    gridtype = type(grid)

    if gridtype==Grid:
        # convert from mtuq object to xarray DataArray
        da = grid.to_dataarray(values)

        _plot_likelihood_regular(filename, da)


    elif gridtype==UnstructuredGrid:
        # convert from mtuq object to xarray DataArray
        df = grid.to_dataframe(values)

        _plot_likelihood_random(filename, df)


def _plot_misfit_regular(filename, da):
    """ Plots regularly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # manipulate DataArray
    da = da.min(dim=('rho', 'kappa', 'sigma', 'h'))

    # plot misfit
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(da.coords['v'], da.coords['w'], da.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.savefig(filename)


def _plot_misfit_random(filename, df):
    """ Plots randomly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    raise NotImplementedError


def _plot_likelihood_regular(filename, da):
    """ Plots regularly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # necessary to avoid floating point errors?
    df['values'] /= df['values'].min()

    # manipulate DataArray
    marginal = da.sum(dim=('rho', 'kappa', 'sigma', 'h'))

    # plot misfit
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(da.coords['v'], da.coords['w'], marginal.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])


def _plot_likelihood_random(filename, df, npts_v=20, npts_w=40):
    """ Plots randomly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # necessary to avoid floating point erros
    df['values'] /= df['values'].min()

    # define edges of cells
    v = closed_interval(-1./3., 1./3., npts_v+1)
    w = closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w+1)

    # define centers of cells
    vp = open_interval(-1./3., 1./3., npts_v)
    wp = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    # sum over parameters to obtain marginal distribution
    marginal = np.empty((npts_w, npts_v))
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(v[_j], v[_j+1]) &
                df['w'].between(w[_i], w[_i+1])]

            # what are the actual and expected number of grid points?
            na = len(subset)
            ne = len(df)/float(npts_v*npts_w)

            marginal[_i, _j] = np.exp(-subset['values']).sum()
            marginal[_i, _j] *= ne/na**2

    # plot misfit
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(vp, wp, marginal)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.savefig(filename)



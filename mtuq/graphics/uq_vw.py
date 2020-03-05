
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
        # convert from mtuq.Grid to xarray.DataArray
        da = grid.to_dataarray(values)

        _plot_misfit_regular(filename, da)

    elif gridtype==UnstructuredGrid:
        # convert from mtuq.UnstructuredGrid to pandas.Dataframe
        df = grid.to_dataframe(values)

        _plot_misfit_random(filename, df)


def plot_likelihood_vw(filename, grid, values=None):
    """ Plots probability density function on 'v-w' rectangle
    (matplotlib implementation)
    """
    gridtype = type(grid)

    if gridtype==Grid:
        # convert from mtuq.Grid to xarray.DataArray
        da = grid.to_dataarray(values)

        _plot_likelihood_regular(filename, da)


    elif gridtype==UnstructuredGrid:
        # convert from mtuq.UnstructuredGrid to pandas.Dataframe
        df = grid.to_dataframe(values)

        _plot_likelihood_random(filename, df)


def _plot_misfit_regular(filename, da):
    """ Plots regularly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # manipulate DataArray
    da = da.min(dim=('rho', 'kappa', 'sigma', 'h'))

    _plot_v_w(da.coords['v'], da.coords['w'], da.values.T)
    pyplot.savefig(filename)


def _plot_misfit_random(filename, df, npts_v=20, npts_w=40):
    """ Plots randomly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # define edges of cells
    v = closed_interval(-1./3., 1./3., npts_v+1)
    w = closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w+1)

    # define centers of cells
    vp = open_interval(-1./3., 1./3., npts_v)
    wp = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    # sum over likelihoods to obtain marginal distribution
    best_misfit = np.empty((npts_w, npts_v))
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(v[_j], v[_j+1]) &
                df['w'].between(w[_i], w[_i+1])]

            best_misfit[_i, _j] = subset['values'].min()

    _plot_v_w(vp, wp, best_misfit)
    pyplot.savefig(filename)


def _plot_likelihood_regular(filename, da):
    """ Plots regularly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # better way to estimate sigma?
    sigma = np.mean(da.values)**0.5
    da.values /= sigma**2.

    # sum over likelihoods to obtain marginal distribution
    da.values = np.exp(-da.values/2.)
    marginal = da.sum(dim=('rho', 'kappa', 'sigma', 'h'))
    marginal /= np.pi/2*marginal.sum()

    _plot_v_w(da.coords['v'], da.coords['w'], marginal.values.T)
    pyplot.savefig(filename)


def _plot_likelihood_random(filename, df, npts_v=20, npts_w=40):
    """ Plots randomly-spaced values on 'v-w' rectangle
    (matplotlib implementation)
    """
    # better way to estimate sigma?
    sigma = np.mean(df['values'])**0.5
    df['values'] /= sigma**2.

    # define edges of cells
    v = closed_interval(-1./3., 1./3., npts_v+1)
    w = closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w+1)

    # define centers of cells
    vp = open_interval(-1./3., 1./3., npts_v)
    wp = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    # sum over likelihoods to obtain marginal distribution
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

            marginal[_i, _j] = np.exp(-subset['values']/2.).sum()
            marginal[_i, _j] *= ne/na**2

    marginal /= np.pi/2*marginal.sum()


    _plot_v_w(vp, wp, marginal)
    pyplot.savefig(filename)


def _plot_v_w(v, w, values):
    pyplot.figure(figsize=(3., 8.))
    pyplot.pcolor(v, w, values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])



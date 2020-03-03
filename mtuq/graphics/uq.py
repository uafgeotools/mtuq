
import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.math import closed_interval, open_interval
from mtuq.util.xarray import dataarray_to_table


def plot_misfit_dc(filename, grid, values):
    """ Plots misfit values over strike, dip, and slip
    (matplotlib implementation)
    """
    da = check_grid('DoubleCouple', grid, values)

    # manipulate DataArray
    da = da.min(dim='rho')

    if 'v' in grid.dims:
        assert len(da.coords['v'])==1
        da = da.squeeze(dim='v')

    if 'w' in grid.dims:
        assert len(da.coords['w'])==1
        da = da.squeeze(dim='w')


    # prepare axes
    fig, axes = pyplot.subplots(2, 2, 
        figsize=(8., 6.),
        )

    pyplot.subplots_adjust(
        wspace=0.33,
        hspace=0.33,
        )

    kwargs = {
        'cmap': 'plasma',
        }

    axes[1][0].axis('off')


    # FIXME: do labels correspond to the correct axes ?!
    marginal = da.min(dim=('sigma'))
    x = marginal.coords['h']
    y = marginal.coords['kappa']
    pyplot.subplot(2, 2, 1)
    pyplot.pcolor(x, y, marginal.values, **kwargs)
    pyplot.xlabel('cos(dip)')
    pyplot.ylabel('strike')

    marginal = da.min(dim=('h'))
    x = marginal.coords['sigma']
    y = marginal.coords['kappa']
    pyplot.subplot(2, 2, 2)
    pyplot.pcolor(x, y, marginal.values, **kwargs)
    pyplot.xlabel('slip')
    pyplot.ylabel('strike')

    marginal = da.min(dim=('kappa'))
    x = marginal.coords['sigma']
    y = marginal.coords['h']
    pyplot.subplot(2, 2, 4)
    pyplot.pcolor(x, y, marginal.values.T, **kwargs)
    pyplot.xlabel('slip')
    pyplot.ylabel('cos(dip)')

    pyplot.savefig(filename)


def plot_misfit_vw(filename, grid, values):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
    da = check_grid('FullMomentTensor', grid, values)

    # manipulate DataArray
    da = da.min(dim='rho')
    da = da.min(dim=('kappa', 'sigma', 'h'))

    # plot misfit
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(vw.coords['v'], vw.coords['w'], vw.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.savefig(filename)


def plot_likelihood_vw(filename, grid, values=None):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
    try:
        da = check_grid('FullMomentTensorRegular', grid, values)
        _plot_likelihood_vw_regular(filename, da)

    except:
        df = check_grid('FullMomentTensorRandom', grid, values)
        _plot_likelihood_vw_random(filename, df)


def _plot_likelihood_vw_regular(filename, da):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
    # manipulate DataArray
    da = da.min(dim='rho')
    da = da.min(dim=('kappa', 'sigma', 'h'))

    # plot misfit
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(vw.coords['v'], vw.coords['w'], vw.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])


def _plot_likelihood_vw_random(filename, df, npts_v=20, npts_w=40):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
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


def check_grid(grid_type, grid, values):
    if grid_type in ('DoubleCouple'):
        _check_dc(grid)

    elif grid_type in ('FullMomentTensor'):
        _check_fmt(grid)

    elif grid_type in ('FullMomentTensorRegular'):
        _check_fmt(grid)

    elif grid_type in ('FullMomentTensorRandom'):
        _check_fmt_random(grid)

    else:
        raise ValueError("Unexpected grid_type")

    if type(grid)==Grid:
        return grid.to_xarray(values)

    elif type(grid)==UnstructuredGrid:
        return grid.to_dataframe(values.flatten())

    else:
        return grid


def _check_dc(grid):
    for dim in ('kappa', 'sigma', 'h'):
        assert dim in grid.dims, Exception("Unexpected grid format")


def _check_fmt(grid):
    if grid.ndim != 6:
        raise Exception("Unexpected grid dimension")

    for dim in ('rho', 'v', 'w', 'kappa', 'sigma', 'h'):
        assert dim in grid.dims, Exception("Unexpected grid format")



def _check_fmt_random(grid):
    for dim in ('rho', 'v', 'w', 'kappa', 'sigma', 'h'):
        pass # assert dim in grid.dims, Exception("Unexpected grid format")

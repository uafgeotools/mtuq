
import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid
from mtuq.util.lune import to_delta, to_gamma
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


def plot_likelihood_vw(filename, grid, values):
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


def check_grid(grid_type, grid, values):
    if grid_type in ('DC', 'dc', 'DoubleCouple'):
        _check_dc(grid)

    elif grid_type in ('FMT', 'fmt', 'FullMomentTensor'):
        _check_fmt(grid)

    else:
        raise ValueError("Unexpected grid_type")

    if type(grid)==Grid:
        return grid.to_xarray(values)
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




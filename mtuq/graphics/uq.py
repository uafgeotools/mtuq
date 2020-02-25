
import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.xarray import dataarray_to_table


def plot_misfit_dc(filename, grid, misfit):
    """ Plots misfit values over strike, dip, and slip
    (matplotlib implementation)
    """
    da = check_grid('DoubleCouple', grid)

    # manipulate DataArray
    da = da.min(dim='rho')
    da = da.min(dim=('kappa', 'sigma', 'h'))

    # plot marginals
    raise NotImplementedError
    pyplot.subplots()

    pyplot.savefig(filename)


def plot_misfit_vw(filename, grid, misfit):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
    da = check_grid('FullMomentTensor', grid)

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


def plot_likelihood_vw(filename, grid, misfit):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
    da = check_grid('FullMomentTensor', grid)

    # manipulate DataArray
    da = da.min(dim='rho')
    da = da.min(dim=('kappa', 'sigma', 'h'))

    # plot misfit
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(vw.coords['v'], vw.coords['w'], vw.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])


def check_grid(grid_type, grid):
    if grid_type in ('DC', 'dc', 'DoubleCouple'):
        _check_dc(grid)

    elif grid_type in ('FMT', 'fmt', 'FullMomentTensor'):
        _check_fmt(grid)

    else:
        raise ValueError("Unexpected grid_type")

    if type(grid)==Grid:
        return grid.as_datarray()
    else:
        return grid


def _check_dc(grid):
    if grid.ndim != 3:
        raise Exception("Unexpected grid dimension")

    for dim in ('kappa', 'sigma', 'h'):
        assert dim in grid.dims, Exception("Unexpected grid format")


def _check_fmt(grid):

    if grid.ndim != 6:
        raise Exception("Unexpected grid dimension")

    for dim in ('rho', 'v', 'w', 'kappa', 'sigma', 'h'):
        assert dim in grid.dims, Exception("Unexpected grid format")




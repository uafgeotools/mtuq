
import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.xarray import dataarray_to_table


def plot_misfit(filename, grid, misfit):
    """ Plots misfit values on lune
    (GMT implementation)
    """
    _check(grid, misfit)
    
    # convert to DataArray for easy manipulation
    da = grid.as_dataarray(misfit)

    da = da.min(['rho', 'kappa', 'sigma', 'h'])
    da.values -= da.values.min()
    da.values /= da.values.max()

    delta = to_delta(da.coords['w'])
    gamma = to_gamma(da.coords['v'])
    
    delta, gamma = np.meshgrid(delta, gamma)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = da.values.flatten() 
    
    np.savetxt('misfit.txt',
        np.column_stack([gamma, delta, values]))
    
    
def plot_likelihood(filename, grid, misfit):
    """ Plots likelihood values on lune
    (GMT implementation)
    """
    _check(grid, misfit)

    # convert to DataArray for easy manipulation
    da = grid.as_dataarray(misfit)

    da.values = np.exp(-da.values)
    da = da.sum(['rho', 'kappa', 'sigma', 'h'])

    delta = to_delta(da.coords['w'])
    gamma = to_gamma(da.coords['v'])

    delta, gamma = np.meshgrid(delta, gamma)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = da.values.flatten()

    np.savetxt('likelihood.txt',
        np.column_stack([gamma, delta, values]))
    

def plot_misfit_vw(filename, grid, misfit):
    """ Plots misfit values on v-w rectangle
    (GMT implementation)
    """
    _check(grid, misfit)

    # convert to DataArray for easy manipulation
    da = grid.as_dataarray(misfit)

    da = da.min(['rho', 'kappa', 'sigma', 'h'])
    da.values -= da.values.min()
    da.values /= da.values.max()

    np.savetxt('misfit_vw.txt', 
        dataarray_to_table(da, ('v', 'w')))


def plot_likelihood_vw(filename, grid, misfit):
    """ Plots likelihood values on v-w rectangle
    (GMT implementation)
    """
    _check(grid, misfit)

    # convert to DataArray for easy manipulation
    da = grid.as_dataarray(misfit)

    da.values = np.exp(-values)
    da = da.sum(['rho', 'kappa', 'sigma', 'h'])
    da.values -= da.values.min()
    da.values /= da.values.max()

    np.savetxt('likelihood_vw.txt', 
        dataarray_to_table(da, ('v', 'w')))


def plot_misfit_vw_matplotlib(filename, grid, misfit):
    """ Plots values on moment tensor grid using v-w projection
    (matplotlib implementation)
    """
    _check(grid, misfit)

    # convert to DataArray for easy manipulation
    da = grid.as_dataarray(misfit)

    da = da.min(dim='rho')
    da = da.min(dim=('kappa', 'sigma', 'h'))

    # now actually plot values
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(vw.coords['v'], vw.coords['w'], vw.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.savefig(filename)


def _check(grid, misfit):
    if type(grid) != Grid:
        raise TypeError

    for dim in ('rho', 'v', 'w', 'kappa', 'sigma', 'h'):
        assert dim in grid.dims, Exception("Unexpected grid format")



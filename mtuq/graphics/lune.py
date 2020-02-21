
import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid
from mtuq.util.lune import to_delta_gamma
from mtuq.util.xarray import dataarray_to_table


def plot_vw(filename, grid, values):

    if type(grid) != Grid:
        raise TypeError

    for dim in ('rho', 'v', 'w', 'kappa', 'sigma', 'h'):
        assert dim in grid.dims, Exception("Unexpected grid format")

    for dim in ('v', 'w'):
        assert grid.sizes[dim]>1, Exception("Unexpected grid format")

    # create DataArray
    da = grid.as_dataarray(values)

    # sum over magnitude
    da = da.sum(dim='rho')

    # reduce over orientation
    da = da.min(dim=('kappa', 'sigma', 'h'))

    # now actually plot values
    pyplot.figure(figsize=(2., 7.07))
    pyplot.pcolor(vw.coords['v'], vw.coords['w'], vw.values)
    pyplot.axis('equal')
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.savefig(filename)


def plot_vw_gmt(filename, grid, values):

    da = grid.as_dataarray(values)

    da = da.sum('rho')
    da = da.min(['kappa', 'sigma', 'h'])

    # normalize values
    da.values -= da.values.min()
    da.values /= da.values.max()

    np.savetxt('tmp.vw', dataarray_to_table(da, ('v', 'w')))


def plot_lune_gmt(filename, grid, values):

    da = grid.as_dataarray(values)

    da = da.sum('rho')
    da = da.min(['kappa', 'sigma', 'h'])

    v = np.array(da.coords['v'])
    w = np.array(da.coords['w'])

    print('v', v.min(), v.max())
    print('w', w.min(), w.max())

    delta, gamma = to_delta_gamma(v, w)

    # normalize values
    da.values -= da.values.min()
    da.values /= da.values.max()

    # flatten values
    delta, gamma = np.meshgrid(delta, gamma)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = da.values.flatten()

    np.savetxt('tmp.lune', np.column_stack(
        [gamma, delta, values]))



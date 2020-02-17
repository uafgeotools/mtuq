
import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid


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


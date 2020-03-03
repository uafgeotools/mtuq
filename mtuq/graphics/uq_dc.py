#
# graphics/uq_dc.py - uncertainty quantification of double couple sources
#

import numpy as np

from matplotlib import pyplot
from xarray import DataArray
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.math import closed_interval, open_interval
from mtuq.util.xarray import dataarray_to_table


def plot_misfit_dc(filename, grid, values):
    """ Plots misfit over strike, dip, and slip
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


def _plot_misfit_regular(filename, da):
    """ Plots regularly-spaced misfit values
    (matplotlib implementation)
    """
    # manipulate DataArray
    da = da.min(dim='rho')

    if 'v' in da.dims:
        assert len(da.coords['v'])==1
        da = da.squeeze(dim='v')

    if 'w' in da.dims:
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



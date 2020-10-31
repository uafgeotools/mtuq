#
# graphics/uq/double_couple.py - uncertainty quantification of double couple sources
#

import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics._gmt import read_cpt
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import fullpath, warn
from mtuq.util.math import closed_interval, open_interval, to_delta, to_gamma


def plot_misfit_dc(filename, ds, title=''):
    """ Plots misfit over strike, dip, and slip
    (matplotlib implementation)
    """
    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        _plot_dc(filename, _squeeze(ds), cmap=cmap_panoply)
        
    elif issubclass(type(ds), DataFrame):
        warn('plot_misfit_dc not implemented for irregularly-spaced grids')


def plot_likelihood_dc(filename, ds, sigma=None, title=''):
    assert sigma is not None

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        ds.values = np.exp(-ds.values/(2.*sigma**2))
        _plot_dc(filename, _squeeze(ds), cmap=cmap_hot)

    elif issubclass(type(ds), DataFrame):
        warn('plot_likelihood_dc not implemented for irregularly-spaced grids')


def plot_marginal_dc():
    raise NotImplementedError


def _squeeze(da):
    if 'origin_idx' in da.dims:
        da = da.max(dim='origin_idx')

    if 'rho' in da.dims:
        da = da.max(dim='rho')

    if 'v' in da.dims:
        assert len(da.coords['v'])==1
        da = da.squeeze(dim='v')

    if 'w' in da.dims:
        assert len(da.coords['w'])==1
        da = da.squeeze(dim='w')

    return da


def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


#
# matplotlib backend
#

def _plot_dc(filename, da, **kwargs):

    # prepare axes
    fig, axes = pyplot.subplots(2, 2, 
        figsize=(8., 8.),
        )

    pyplot.subplots_adjust(
        wspace=0.33,
        hspace=0.33,
        )

    axes[1][0].axis('off')

    # FIXME: do labels correspond to the correct axes ?!
    marginal = da.min(dim=('sigma'))
    x = marginal.coords['h']
    y = marginal.coords['kappa']
    pyplot.subplot(2, 2, 1)
    pyplot.pcolor(x, y, marginal.values, **kwargs)
    pyplot.xlabel('Dip')
    pyplot.xticks(**theta_tick_kwargs)
    pyplot.ylabel('Strike')
    pyplot.yticks(**kappa_tick_kwargs)

    marginal = da.min(dim=('h'))
    x = marginal.coords['sigma']
    y = marginal.coords['kappa']
    pyplot.subplot(2, 2, 2)
    pyplot.pcolor(x, y, marginal.values, **kwargs)
    pyplot.xlabel('Slip')
    pyplot.xticks(**sigma_tick_kwargs)
    pyplot.ylabel('Strike')
    pyplot.yticks(**kappa_tick_kwargs)

    marginal = da.min(dim=('kappa'))
    x = marginal.coords['sigma']
    y = marginal.coords['h']
    pyplot.subplot(2, 2, 4)
    pyplot.pcolor(x, y, marginal.values.T, **kwargs)
    pyplot.xlabel('Slip')
    pyplot.xticks(**sigma_tick_kwargs)
    pyplot.ylabel('Dip')
    pyplot.yticks(**theta_tick_kwargs)
    pyplot.savefig(filename)


kappa_tick_kwargs = {
    'ticks': [0, 45, 90, 135, 180, 225, 270, 315, 360],
    'labels': ['0', '', '90', '', '180', '', '270', '', '360']
}

sigma_tick_kwargs = {
    'ticks': [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90],
    'labels': ['-90', '', '-45', '', '0', '', '45', '', '90'],
}

theta_tick_kwargs = {
    'ticks': [np.cos(np.radians(tick)) for tick in [0, 15, 30, 45, 60, 75, 90]],
    'labels': ['0', '', '30', '', '60', '', '90']
}

try:
    cmap_hot = read_cpt(fullpath('mtuq/graphics/_gmt/cpt/hot.cpt'))
    cmap_hot_r = read_cpt(fullpath('mtuq/graphics/_gmt/cpt/hot_r.cpt'))
    cmap_panoply = read_cpt(fullpath('mtuq/graphics/_gmt/cpt/panoply.cpt'))
except:
    pass

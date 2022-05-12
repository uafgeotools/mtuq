#
# graphics/uq/double_couple.py - uncertainty quantification of double couple sources
#

import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics._gmt import read_cpt, _cpt_path
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import dataarray_idxmin, dataarray_idxmax, fullpath, warn
from mtuq.util.math import closed_interval, open_interval, to_delta, to_gamma, to_mij
from os.path import exists


def plot_misfit_dc(filename, ds, **kwargs):
    """ Plots misfit values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    _defaults(kwargs, {
        'colormap': 'viridis',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        misfit = _misfit_dc_regular(ds)
        
    elif issubclass(type(ds), DataFrame):
        warn('plot_misfit_dc not implemented yet for irregularly-spaced grids.\n'
             'No figure will be generated.')
        return

    _plot_dc(filename, misfit, **kwargs)



def plot_likelihood_dc(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

   ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        likelihoods = _likelihoods_dc_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        warn('plot_misfit_dc not implemented for irregularly-spaced grids. '
             'No figure will be generated.')
        return

    _plot_dc(filename, likelihoods, **kwargs)


def plot_marginal_dc():
    raise NotImplementedError



def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


def _defaults(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
           kwargs[key] = defaults[key]


#
# matplotlib backend
#

def _plot_dc(filename, da, show_best=True, colormap='hot', **kwargs):

    """ Plots DataArray values on vw rectangle

    .. rubric :: Keyword arguments

    ``colormap`` (`str`)
    Color palette used for plotting values 
    (choose from GMT or MTUQ built-ins)

    ``show_best`` (`bool`):
    Show where best-fitting orientation falls on strike, dip, slip plots

    ``title`` (`str`)
    Optional figure title

    ``backend`` (`function`)
    Choose from `_plot_dc_matplotlib` (default) or user-supplied function

    """

    if not issubclass(type(da), DataArray):
        raise Exception()

    if show_best:
        if 'best_dc' in da.attrs:
            best_dc = da.attrs['best_dc']
        else:
            warn("Best-fitting orientation not given")


    # FIXME: do labels correspond to the correct axes ?!

    # prepare axes
    fig, axes = pyplot.subplots(2, 2, 
        figsize=(8., 8.),
        )

    pyplot.subplots_adjust(
        wspace=0.4,
        hspace=0.4,
        )

    if exists(_cpt_path(colormap)):
       colormap = read_cpt(_cpt_path(colormap))

    # upper left panel
    marginal = da.min(dim=('sigma'))
    x = marginal.coords['h']
    y = marginal.coords['kappa']

    axis = axes[0][0]
    _pcolor(axis, x, y, marginal.values, colormap)

    axis.set_xlabel('Dip', **axis_label_kwargs)
    axis.set_xticks(theta_ticks)
    axis.set_xticklabels(theta_ticklabels)

    axis.set_ylabel('Strike', **axis_label_kwargs)
    axis.set_yticks(kappa_ticks)
    axis.set_yticklabels(kappa_ticklabels)

    # upper right panel
    marginal = da.min(dim=('h'))
    x = marginal.coords['sigma']
    y = marginal.coords['kappa']

    axis = axes[0][1]
    _pcolor(axis, x, y, marginal.values, colormap)

    axis.set_xlabel('Slip', **axis_label_kwargs)
    axis.set_xticks(sigma_ticks)
    axis.set_xticklabels(sigma_ticklabels)

    axis.set_ylabel('Strike', **axis_label_kwargs)
    axis.set_yticks(kappa_ticks)
    axis.set_yticklabels(kappa_ticklabels)

    # lower right panel
    marginal = da.min(dim=('kappa'))
    y = marginal.coords['h']
    x = marginal.coords['sigma']

    axis = axes[1][1]
    _pcolor(axis, x, y, marginal.values.T, colormap)

    axis.set_xlabel('Slip', **axis_label_kwargs)
    axis.set_xticks(sigma_ticks)
    axis.set_xticklabels(sigma_ticklabels)

    axis.set_ylabel('Dip', **axis_label_kwargs)
    axis.set_yticks(theta_ticks)
    axis.set_yticklabels(theta_ticklabels)

    # lower left panel
    axes[1][0].axis('off')

    # optional markers
    if show_best:
        _kappa, _sigma, _h = best_dc
        _add_marker(axes[0][0], (_h, _kappa))
        _add_marker(axes[0][1], (_sigma, _kappa))
        _add_marker(axes[1][1], (_sigma, _h))

    pyplot.savefig(filename)


def _add_marker(axis, coords):
    axis.scatter(*coords, s=250,
        marker='o',
        facecolors='none',
        edgecolors=[0,1,0],
        linewidths=1.75,
        clip_on=False,
        zorder=100,
        )


axis_label_kwargs = {
    'fontsize': 14
    }


def _pcolor(axis, x, y, values, colormap, **kwargs):
    # workaround matplotlib compatibility issue
    try:
        axis.pcolor(x, y, values, cmap=colormap, shading='auto',  **kwargs)
    except:
        axis.pcolor(x, y, values, cmap=colormap, **kwargs)


kappa_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]
kappa_ticklabels = ['0', '', '90', '', '180', '', '270', '', '360']

sigma_ticks = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
sigma_ticklabels = ['-90', '', '-45', '', '0', '', '45', '', '90']

theta_ticks = [np.cos(np.radians(tick)) for tick in [0, 15, 30, 45, 60, 75, 90]]
theta_ticklabels = ['0', '', '30', '', '60', '', '90']



#
# for extracting misfit, variance reduction and likelihood from
# regularly-spaced grids
#

def _misfit_dc_regular(da):
    """ For each double couple, extract minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'rho', 'v', 'w'))

    return misfit.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        })


def _likelihoods_dc_regular(da, var):
    """ For each double couple, calculate maximum likelihood
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'rho', 'v', 'w'))
    likelihoods.values /= likelihoods.values.sum()
    #likelihoods /= dc_area

    return likelihoods.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        'maximum_likelihood_estimate': dataarray_idxmax(likelihoods).values(),
        })


def _marginals_dc_regular(da, var):
    """ For each double couple, calculate marginal likelihood
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'rho', 'v', 'w'))
    likelihoods.values /= likelihoods.values.sum()
    likelihoods /= vw_area

    return likelihoods.assign_attrs({
        'best_mt': _min_mt(da),
        'maximum_likelihood_estimate': dataarray_idxmax(likelihoods).values(),
        })


def _min_mt(da):
    """ Returns moment tensor vector corresponding to mininum DataArray value
    """
    da = dataarray_idxmin(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _max_mt(da):
    """ Returns moment tensor vector corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _min_dc(da):
    """ Returns v,w coordinates corresponding to mininum DataArray value
    """
    da = dataarray_idxmin(da)
    dc_keys = ['kappa', 'sigma', 'h']
    dc_vals = [da[key].values for key in dc_keys]
    return dc_vals

def _max_dc(da):
    """ Returns v,w coordinates corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    dc_keys = ['kappa', 'sigma', 'h']
    dc_vals = [da[key].values for key in dc_keys]
    return dc_vals




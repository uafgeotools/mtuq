#
# graphics/uq/double_couple.py - uncertainty quantification of double couple sources
#

import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics._gmt import read_cpt, _cpt_path
from mtuq.graphics.uq._matplotlib import _plot_dc_matplotlib
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import dataarray_idxmin, dataarray_idxmax, defaults, warn
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
    defaults(kwargs, {
        'colormap': 'viridis',
        'squeeze': 'min',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        misfit = _misfit_dc_regular(ds)
        
    elif issubclass(type(ds), DataFrame):
        misfit = _misfit_dc_random(ds)

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
    defaults(kwargs, {
        'colormap': 'hot_r',
        'squeeze': 'max',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        likelihoods = _likelihoods_dc_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = _likelihoods_dc_random(ds, var)

    _plot_dc(filename, likelihoods, **kwargs)



def plot_marginal_dc(filename, ds, var, **kwargs):
    """ Plots marginal likelihood values over strike, dip, slip

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
    defaults(kwargs, {
        'colormap': 'hot_r',
        'squeeze': 'max',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        marginals = _marginals_dc_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = _marginals_dc_random(ds, var)

    _plot_dc(filename, marginals, **kwargs)



def plot_variance_reduction_dc(filename, ds, data_norm, **kwargs):
    """ Plots variance reduction values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

   ``data_norm`` (`float`):
    Data norm


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    defaults(kwargs, {
        'colormap': 'viridis_r',
        'squeeze': 'max',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        variance_reduction = _variance_reduction_dc_regular(ds, data_norm)

    elif issubclass(type(ds), DataFrame):
        variance_reduction = _variance_reduction_dc_random(ds, data_norm)

    _plot_dc(filename, variance_reduction, **kwargs)



def _plot_dc(filename, da, show_best=True, backend=_plot_dc_matplotlib,
    squeeze='min', **kwargs):

    """ Plots DataArray values over strike, dip, slip

    .. rubric :: Keyword arguments

    ``colormap`` (`str`)
    Color palette used for plotting values 
    (choose from GMT or MTUQ built-ins)

    ``show_best`` (`bool`):
    Show where best-fitting moment tensor falls in terms of strike, dip, slip

    ``squeeze`` (`str`):
    By default, 2-D surfaces are obtained by minimizing or maximizing.
    For slices instead, use `slice_min` or `slice_max`.

    ``backend`` (`function`):
    Choose from `_plot_dc_matplotlib` (default) or user-supplied function

    """

    if not issubclass(type(da), DataArray):
        raise Exception()

    if show_best:
        if 'best_dc' in da.attrs:
            best_dc = da.attrs['best_dc']
        else:
            warn("Best-fitting orientation not given")
            best_dc = None

    # note the following parameterization details
    #     kappa = strike
    #     sigma = slip
    #     h = cos(dip)

    # squeeze full 3-D array into 2-D arrays
    if squeeze=='min':
        values_h_kappa = da.min(dim=('sigma')).values
        values_sigma_kappa = da.min(dim=('h')).values
        values_sigma_h = da.min(dim=('kappa')).values.T

    elif squeeze=='max':
        values_h_kappa = da.max(dim=('sigma')).values
        values_sigma_kappa = da.max(dim=('h')).values
        values_sigma_h = da.max(dim=('kappa')).values.T

    elif squeeze=='slice_min':
        argmin = da.argmin(('kappa','sigma','h'))
        values_h_kappa = da.isel(sigma=argmin['sigma'], drop=True).values
        values_sigma_kappa = da.isel(h=argmin['h'], drop=True).values
        values_sigma_h = da.isel(kappa=argmin['kappa'], drop=True).values.T

    elif squeeze=='slice_max':
        argmax = da.argmax(('kappa','sigma','h'))
        values_h_kappa = da.isel(sigma=argmax['sigma'], drop=True).values
        values_sigma_kappa = da.isel(h=argmax['h'], drop=True).values
        values_sigma_h = da.isel(kappa=argmax['kappa'], drop=True).values.T

    else:
        raise ValueError

    backend(filename,
        da.coords,
        values_h_kappa,
        values_sigma_kappa,
        values_sigma_h,
        best_dc=best_dc,
        **kwargs)


#
# for extracting misfit, variance reduction and likelihood from
# regularly-spaced grids
#

def _misfit_dc_regular(da):
    """ For each moment tensor orientation, extract minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'rho', 'v', 'w'))

    return misfit.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        })

def _misfit_dc_random(df, **kwargs):
    df = df.copy()
    df = df.reset_index()

    # Ensure 'misfit' column exists
    if 'misfit' not in df.columns:
        df['misfit'] = df.iloc[:, -1]

    da = _bin_dc_regular(df, lambda group: group['misfit'].min(), **kwargs)

    return da.assign_attrs({
        'best_dc': _min_dc(da),
    })


def _likelihoods_dc_regular(da, var):
    """ For each moment tensor orientation, calculate maximum likelihood
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

def _likelihoods_dc_random(df, var, **kwargs):
    """
    Calculate max likelihood from random dataset bins, ensuring global normalization.
    """

    likelihoods = df.copy().reset_index()

    # Ensure 'misfit' column exists
    if 'misfit' not in likelihoods.columns:
        likelihoods.rename(columns={likelihoods.columns[-1]: 'misfit'}, inplace=True)

    # Compute likelihoods globally and normalize BEFORE binning
    likelihoods['likelihood'] = np.exp(-likelihoods['misfit'] / (2. * var))
    likelihoods['likelihood'] /= likelihoods['likelihood'].sum()  # Global normalization

    # Apply binning, operating on globally normalized likelihoods
    binned_likelihoods = _bin_dc_regular(
        likelihoods, lambda group: group['likelihood'].max(), **kwargs
    )

    return binned_likelihoods.assign_attrs({
        'best_dc': _max_dc(binned_likelihoods),
        'maximum_likelihood_estimate': dataarray_idxmax(binned_likelihoods).values(),
    })


def _marginals_dc_regular(da, var):
    """ For each moment tensor orientation, calculate marginal likelihood
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    marginals = likelihoods.sum(dim=('origin_idx', 'rho', 'v', 'w'))
    marginals.values /= marginals.values.sum()

    return marginals.assign_attrs({
        'best_dc': _max_dc(marginals),
        })

def _marginals_dc_random(df, var, **kwargs):
    """
    Calculate marginal likelihoods for random bins with global normalization.
    """

    likelihoods = df.copy().reset_index()

    if 'misfit' not in likelihoods.columns:
        likelihoods.rename(columns={likelihoods.columns[-1]: 'misfit'}, inplace=True)

    # Compute likelihoods and normalize globally
    likelihoods['likelihood'] = np.exp(-likelihoods['misfit'] / (2. * var))
    likelihoods['likelihood'] /= likelihoods['likelihood'].sum()  # Global normalization

    # Sum within bins after global normalization
    marginals = _bin_dc_regular(
        likelihoods, lambda group: group['likelihood'].sum(), **kwargs
    )

    # No need for further normalization, already globally adjusted
    return marginals.assign_attrs({
        'best_dc': _max_dc(marginals),
    })


def _variance_reduction_dc_regular(da, data_norm):
    """ For each moment tensor orientation, extracts maximum variance reduction
    """
    variance_reduction = 1. - da.copy()/data_norm

    variance_reduction = variance_reduction.max(
        dim=('origin_idx', 'rho', 'v', 'w'))

    # widely-used convention - variance reducation as a percentage
    variance_reduction.values *= 100.

    return variance_reduction.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        })

def _variance_reduction_dc_random(df, data_norm, **kwargs):
    df = df.copy()
    df = df.reset_index()

    # Ensure 'misfit' column exists
    if 'misfit' not in df.columns:
        df['misfit'] = df.iloc[:, -1]

    da = _bin_dc_regular(df, lambda group: 1. - group['misfit'].min()/data_norm, **kwargs)

    return da.assign_attrs({
        'best_dc': _max_dc(da),
    })


#
# utility functions
#

def _min_mt(da):
    """ Returns moment tensor vector corresponding to minimum DataArray value
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
    """ Returns orientation angles corresponding to minimum DataArray value
    """
    da = dataarray_idxmin(da)
    dc_keys = ['kappa', 'sigma', 'h']
    dc_vals = [da[key].values for key in dc_keys]
    return dc_vals

def _max_dc(da):
    """ Returns orientation angles corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    dc_keys = ['kappa', 'sigma', 'h']
    dc_vals = [da[key].values for key in dc_keys]
    return dc_vals


def _bin_dc_regular(df, handle, npts=25, **kwargs):
    """Bins irregularly-spaced moment tensor orientations into regular grids for plotting."""
    # Orientation bins
    kappa_min, kappa_max = 0, 360
    sigma_min, sigma_max = -90, 90
    h_min, h_max = 0, 1

    kappa_edges = np.linspace(kappa_min, kappa_max, npts + 1)
    sigma_edges = np.linspace(sigma_min, sigma_max, npts + 1)
    h_edges = np.linspace(h_min, h_max, npts + 1)

    kappa_centers = 0.5 * (kappa_edges[:-1] + kappa_edges[1:])
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])

    # Prepare the data arrays
    kappa_vals = df['kappa'].values
    sigma_vals = df['sigma'].values
    h_vals = df['h'].values

    # Compute bin indices for each data point
    kappa_indices = np.digitize(kappa_vals, kappa_edges) - 1
    sigma_indices = np.digitize(sigma_vals, sigma_edges) - 1
    h_indices = np.digitize(h_vals, h_edges) - 1

    # Ensure indices are within valid range
    kappa_indices = np.clip(kappa_indices, 0, npts - 1)
    sigma_indices = np.clip(sigma_indices, 0, npts - 1)
    h_indices = np.clip(h_indices, 0, npts - 1)

    # Add bin indices to DataFrame
    df = df.copy()
    df['kappa_idx'] = kappa_indices
    df['sigma_idx'] = sigma_indices
    df['h_idx'] = h_indices

    # Group by bin indices
    # grouped = df.groupby(['h_idx', 'sigma_idx', 'kappa_idx'])
    grouped = df.groupby(['kappa_idx', 'sigma_idx', 'h_idx'])

    # Initialize the output array with appropriate data type
    binned = np.full((npts, npts, npts), np.nan)

    # Process each group
    for (k_idx, s_idx, h_idx), group in grouped:
        # Apply the handle function to the group DataFrame
        binned[k_idx, s_idx, h_idx] = handle(group)

    # Create the DataArray
    da = DataArray(
        data=binned,
        dims=('kappa', 'sigma', 'h'),
        coords={'kappa': kappa_centers, 'sigma': sigma_centers, 'h': h_centers},
    )

    return da

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")



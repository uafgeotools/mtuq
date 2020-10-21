
#
# graphics/uq/lune.py - uncertainty quantification on the eigenvalue lune
#
# For details about the eigenvalue lune, see 
# Tape2012 - A geometric setting for moment tensors
# (https://doi.org/10.1111/j.1365-246X.2012.05491.x)
#

import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics.uq._gmt import exists_gmt, gmt_not_found_warning, \
    gmt_plot_misfit_lune, gmt_plot_likelihood_lune
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util.math import lune_det, to_gamma, to_delta, to_v, to_w, semiregular_grid


def plot_misfit_lune(filename, ds, 
    title='',
    callback=None,
    add_colorbar=True, 
    add_marker=True,
    colorbar_label=None, 
    show_beachballs=False):

    """ Plots misfit values on eigenvalue lune (requires GMT)


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title

    """
    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        ds = ds.min(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(ds.coords['v'])
        delta = to_delta(ds.coords['w'])
        values = ds.values.transpose()

    elif issubclass(type(ds), DataFrame):
        ds = ds.reset_index()
        gamma, delta, values = _bin(ds, lambda ds: ds.min())

    if callback:
        values = callback()

    gmt_plot_misfit_lune(filename, gamma, delta, values, 
        add_colorbar=add_colorbar, add_marker=add_marker, title=title)



def plot_likelihood_lune(filename, ds, sigma=None, 
    add_colorbar=True, add_marker=True, title=''):
    """ Plots maximum likelihoods on eigenvalue lune (requires GMT)


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``sigma`` (`float`):
    Standard deviation applied to misfit values

    ``title`` (`str`):
    Optional figure title

    """
    assert sigma is not None

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        ds.values = np.exp(-ds.values/(2.*sigma**2))
        ds = ds.max(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(ds.coords['v'])
        delta = to_delta(ds.coords['w'])
        values = ds.values.transpose()

    elif issubclass(type(ds), DataFrame):
        ds = np.exp(-ds/(2.*sigma**2))
        ds = ds.reset_index()
        gamma, delta, values = _bin(ds, lambda ds: ds.max())

    values /= values.sum()

    gmt_plot_likelihood_lune(filename, gamma, delta, values,
        add_colorbar=add_colorbar, add_marker=add_marker, title=title)


def plot_marginal_lune(filename, ds, sigma=None,
    add_colorbar=True, add_marker=True, title=''):
    """ Plots marginal likelihoods on eigenvalue lune (requires GMT)
    
    
    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``sigma`` (`float`):
    Standard deviation applied to misfit values
        
    ``title`` (`str`):
    Optional figure title

    """
    assert sigma is not None

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        ds.values = np.exp(-ds.values/(2.*sigma**2))
        ds = ds.sum(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(ds.coords['v'])
        delta = to_delta(ds.coords['w'])
        values = ds.values.transpose()

    elif issubclass(type(ds), DataFrame):
        ds = np.exp(-ds/(2.*sigma**2))
        ds = ds.reset_index()
        gamma, delta, values = _bin(ds, lambda ds: ds.sum()/len(ds), normalize=True)

    #values /= lune_det(delta, gamma)
    values /= values.sum()

    gmt_plot_likelihood_lune(filename, gamma, delta, values,
        add_colorbar=add_colorbar, add_marker=add_marker, title=title)



# utility functions

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


def _bin(df, handle, npts_v=20, npts_w=40, tightness=0.6, normalize=False):
    """ Bins DataFrame into rectangular cells
    """
    # at which points will we plot values?
    centers_v, centers_w = semiregular_grid(
        npts_v, npts_w, tightness=tightness)

    # what cell edges correspond to the above centers?
    centers_gamma = to_gamma(centers_v)
    edges_gamma = np.array(centers_gamma[:-1] + centers_gamma[1:])/2.
    edges_v = to_v(edges_gamma)

    centers_delta = to_delta(centers_w)
    edges_delta = np.array(centers_delta[:-1] + centers_delta[1:])/2.
    edges_w = to_w(edges_delta)

    edges_v = np.pad(edges_v, 1)
    edges_v[0] = -1./3.
    edges_v[-1] = +1./3.

    edges_w = np.pad(edges_w, 1)
    edges_w[0] = -3.*np.pi/8.
    edges_w[-1] = +3.*np.pi/8


    # bin grid points into cells
    binned = np.empty((npts_w, npts_v))
    binned[:] = np.nan
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(edges_v[_j], edges_v[_j+1]) &
                df['w'].between(edges_w[_i], edges_w[_i+1])]

            if len(subset)==0:
                print("Encountered empty bin")

            binned[_i, _j] = handle(subset[0])

            if normalize:
              # normalize by area of cell
              binned[_i, _j] /= edges_v[_j+1] - edges_v[_j]
              binned[_i, _j] /= edges_w[_i+1] - edges_w[_i]

    return to_gamma(centers_v), to_delta(centers_w), binned



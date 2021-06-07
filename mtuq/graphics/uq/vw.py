
#
# For details about the v,w rectangle see 
# Tape2015 - A uniform parameterization of moment tensors
# (https://doi.org/10.1093/gji/ggv262)
#

import numpy as np
import pandas
import xarray

from matplotlib import pyplot
from mtuq.grid_search import DataArray, DataFrame, MTUQDataArray, MTUQDataFrame
from mtuq.graphics._gmt import read_cpt
from mtuq.graphics.uq._gmt import _nothing_to_plot
from mtuq.util import dataarray_idxmin, dataarray_idxmax, fullpath
from mtuq.util.math import closed_interval, open_interval, semiregular_grid,\
    to_v, to_w, to_gamma, to_delta, to_mij
from os.path import exists


v_min = -1./3.
v_max = +1./3.
w_min = -3.*np.pi/8.
w_max = +3.*np.pi/8.
vw_area = (v_max-v_min)*(w_max-w_min)



def plot_misfit_vw(filename, ds, **kwargs):
    """ Plots misfit values on v,w rectangle

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    See _plot_vw for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'viridis',
        'marker_type': 1,
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        misfit = calculate_misfit(ds)

    elif issubclass(type(ds), DataFrame):
        misfit = calculate_misfit_unstruct(ds)

    _plot_vw(filename, misfit, **kwargs)


def plot_likelihood_vw(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values on v,w rectangle

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    See _plot_vw for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        'marker_type': 2,
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        likelihoods = calculate_likelihoods(ds, var)

    elif issubclass(type(ds), DataFrame):
        likelihoods = calculate_likelihoods_unstruct(ds, var)

    _plot_vw(filename, likelihoods, **kwargs)


def plot_marginal_vw(filename, ds, var, **kwargs):
    """ Plots marginal likelihoods on v,w rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance


    See _plot_vw for keyword argument descriptions

    """
    _defaults(kwargs, {
        'colormap': 'hot_r',
        'marker_type': 2,
        })

    _check(ds)
    ds = ds.copy()

    if issubclass(type(ds), DataArray):
        marginals = calculate_marginals(ds, var)

    elif issubclass(type(ds), DataFrame):
        marginals = calculate_marginals_unstruct(ds, var)

    _plot_vw(filename, marginals, **kwargs)



#
# matplotlib backend
#

def _plot_vw(filename, da,
    colormap='viridis', colorbar_type=1, marker_type=1, title=''):

    if marker_type not in [0,1,2]:
        raise ValueError

    v = da.coords['v']
    w = da.coords['w']
    values = da.values

    if _nothing_to_plot(values):
        return

    fig, ax = pyplot.subplots(figsize=(3., 8.), constrained_layout=True)

    # pcolor requires corners of pixels
    corners_v = _centers_to_edges(v)
    corners_w = _centers_to_edges(w)

    # `values` gets mapped to pixel colors
    pyplot.pcolor(corners_v, corners_w, values.transpose(), cmap=colormap)

    # v and w have the following bounds
    # (see https://doi.org/10.1093/gji/ggv262)
    pyplot.xlim([-1./3., 1./3.])
    pyplot.ylim([-3./8.*np.pi, 3./8.*np.pi])

    pyplot.xticks([], [])
    pyplot.yticks([], [])

    if exists(_local_path(colormap)):
       cmap = read_cpt(_local_path(colormap))

    if colorbar_type:
        cbar = pyplot.colorbar(
            orientation='horizontal',
            pad=0.,
            )

        cbar.formatter.set_powerlimits((-2, 2))

    if title:
        fontdict = {'fontsize': 16}
        pyplot.title(title, fontdict=fontdict)

    if marker_type > 0:
        if marker_type==1:
            idx = np.unravel_index(da.values.argmin(), da.values.shape)
            coords = v[idx[0]], w[idx[1]]
        elif marker_type==2:
            idx = np.unravel_index(da.values.argmax(), da.values.shape)
            coords = v[idx[0]], w[idx[1]]

        pyplot.scatter(*coords, s=333,
            marker='o',
            facecolors='none',
            edgecolors=[0,1,0],
            linewidths=1.75,
            )

    pyplot.savefig(filename)
    pyplot.close()


#
# for extracting misfit or likelihood from regularly-spaced grids
#

def calculate_misfit(da):
    """ For each point on lune, extracts minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))

    return misfit.assign_attrs({
        'best_mt': _best_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        })


def calculate_likelihoods(da, var):
    """ For each point on lune, calculates maximum likelihood value
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
    likelihoods.values /= likelihoods.values.sum()
    likelihoods /= vw_area

    return likelihoods.assign_attrs({
        'best_mt': _best_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        'likelihood_max': likelihoods.max(),
        'likelihood_vw': dataarray_idxmax(likelihoods).values(),
        })


def calculate_marginals(da, var):
    """ For each point on lune, calculates marginal likelihood value
    """

    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    marginals = likelihoods.sum(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
    marginals.values /= marginals.values.sum()
    marginals /= vw_area

    return marginals.assign_attrs({
        'best_mt': _best_mt(da),
        'best_vw': _min_vw(da),
        'lune_array': _lune_array(da),
        'marginal_max': marginals.max(),
        'marginal_vw': dataarray_idxmax(marginals).values(),
        })


def _lune_array(da):
    """ For each point on lune, collects best-fitting moment tensor
    """
    #
    # TODO - generalize for multiple orgins
    #
    origin_idx = 0

    nv = len(da.coords['v'])
    nw = len(da.coords['w'])

    lune_array = np.empty((nv*nw,6))
    for iv in range(nv):
        for iw in range(nw):
            sliced = da[:,iv,iw,:,:,:,origin_idx]
            argmin = np.argmin(sliced.values, axis=None)
            idx = np.unravel_index(argmin, np.shape(sliced))

            lune_array[nw*iv+iw,0] = da['rho'][idx[0]]
            lune_array[nw*iv+iw,1] = da['v'][iv]
            lune_array[nw*iv+iw,2] = da['w'][iw]
            lune_array[nw*iv+iw,3] = da['kappa'][idx[1]]
            lune_array[nw*iv+iw,4] = da['sigma'][idx[2]]
            lune_array[nw*iv+iw,5] = da['h'][idx[3]]

    return lune_array


def _best_mt(da):
    """ Returns overall best-fitting moment tensor
    """
    da = dataarray_idxmin(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _min_vw(da):
    """ Returns overall best v,w
    """
    da = dataarray_idxmin(da)
    lune_keys = ['v', 'w']
    lune_vals = [da[key].values for key in lune_keys]
    return lune_vals

def _max_vw(da):
    """ Returns overall best v,w
    """
    da = dataarray_idxmax(da)
    lune_keys = ['v', 'w']
    lune_vals = [da[key].values for key in lune_keys]
    return lune_vals



#
# for extracting misfit or likelihood from irregularly-spaced grids
#

def calculate_misfit_unstruct(df, **kwargs):
    df = df.copy()
    df = df.reset_index()
    da = vw_bin_semiregular(df, lambda df: df.min(), **kwargs)

    return da.assign_attrs({
        'best_vw':  _min_vw(da),
        })


def calculate_likelihoods_unstruct(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = vw_bin_semiregular(df, lambda df: df.max(), **kwargs)
    da.values /= da.values.sum()
    da.values /= vw_area

    return da.assign_attrs({
        'likelihood_max': da.max(),
        'likelihood_vw': _max_vw(da),
        'best_vw': _max_vw(da),
        })


def calculate_marginals_unstruct(df, var, **kwargs):
    df = df.copy()
    df = np.exp(-df/(2.*var))
    df = df.reset_index()

    da = vw_bin_semiregular(df, lambda df: df.sum()/len(df))
    da.values /= da.values.sum()
    da.values /= vw_area

    return da.assign_attrs({
        'marginal_max': da.max(),
        'marginal_vw': _max_vw(da),
        })


#
# bins irregularly-spaced moment tensors into v,w rectangles
#

def vw_bin_regular(df, handle, npts_v=20, npts_w=40):
    """ Bins irregularly-spaced moment tensors into square v,w cells
    """
    # define centers of cells
    centers_v = open_interval(-1./3., 1./3., npts_v)
    centers_w = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    # define corners of cells
    v = closed_interval(-1./3., 1./3., npts_v+1)
    w = closed_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w+1)

    binned = np.empty((npts_w, npts_v))
    for _i in range(npts_w):
        for _j in range(npts_v):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['v'].between(v[_j], v[_j+1]) &
                df['w'].between(w[_i], w[_i+1])]

            binned[_i, _j] = handle(subset[0])

    return DataArray(
        dims=('v', 'w'),
        coords=(centers_v, centers_w),
        data=binned.transpose()
        )


def vw_bin_semiregular(df, handle, npts_v=20, npts_w=40, tightness=0.6, normalize=False):
    """ Bins irregularly-spaced moment tensors into rectangular v,w cells
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

    return DataArray(
        dims=('v', 'w'),
        coords=(centers_v, centers_w),
        data=binned.transpose()
        )


def _centers_to_edges(v):
    if issubclass(type(v), DataArray):
        v = v.values.copy()
    else:
        v = v.copy()

    dv = (v[1]-v[0])
    v -= dv/2
    v = np.pad(v, (0, 1))
    v[-1] = v[-2] + dv

    return v



#
# utility functions
#

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


def _defaults(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
           kwargs[key] = defaults[key]

def _local_path(name):
    return fullpath('mtuq/graphics/_gmt/cpt', name+'.cpt')



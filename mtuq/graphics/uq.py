
#
# graphics/uq.py - uncertainty quantification on the eigenvalue lune
#
# For details about the eigenvalue lune, see 
# Tape2012 - A geometric setting for moment tensors
# (https://doi.org/10.1111/j.1365-246X.2012.05491.x)
#


import numpy as np
import shutil
import subprocess
import warnings

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics._gmt import gmt_cmd, check_ext
from mtuq.util import fullpath
from mtuq.util.lune import to_gamma, to_delta, to_v, to_w, semiregular_grid


def plot_misfit(filename, ds, title=None):
    """ Plots misfit values on eigenvalue lune (requires GMT)


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)


    .. note ::

      This utility requires Generic Mapping Tools >=5.

      To display information about supported image formats:
      `gmt psconvert --help`.

      For a matplotlib-only alternative: `mtuq.graphics.plot_misfit_vw`.


    """
    ds = ds.copy()


    if issubclass(type(ds), DataArray):
        da = ds
        da = da.min(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        gamma, delta, values = _bin(df, lambda df: df.min())


    _plot_lune(filename, gamma, delta, values, title)



def plot_likelihood(filename, ds, sigma=1., title=None):
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


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)


    .. note ::

      This utility requires Generic Mapping Tools >=5.

      To display information about supported image formats:
      `gmt psconvert --help`.

      For a matplotlib-only alternative: `mtuq.graphics.plot_misfit_vw`.

    """
    ds = ds.copy()


    # convert from misfit to likelihood
    ds.values = np.exp(-ds.values/(2.*sigma**2))


    if issubclass(type(ds), DataArray):
        da = ds
        da = da.max(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        gamma, delta, values = _bin(df, lambda df: df.max())


    values /= values.sum()

    _plot_lune(filename, gamma, delta, values, title)



def plot_marginal(filename, ds, sigma=1., title=None):
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


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)

        
    .. note ::

      This utility requires Generic Mapping Tools >=5.

      To display information about supported image formats:
      `gmt psconvert --help`.

      For a matplotlib-only alternative: `mtuq.graphics.plot_misfit_vw`.
 
    """

    ds = ds.copy()


    # convert from misfit to likelihood
    ds.values = np.exp(-ds.values/(2.*sigma**2))


    if issubclass(type(ds), DataArray):
        da = ds
        da = da.sum(dim=('origin_idx', 'rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        gamma, delta, values = _bin(df, lambda df: df.sum()/len(df), normalize=True)

    values /= lune_det(delta, gamma)
    values /= values.sum()

    _plot_lune(filename, gamma, delta, values)



#
# utilities for irregularly-spaced grids
#

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



#
# GMT wrappers
#

def _plot_lune(filename, gamma, delta, values, title=None):
    """ Plots misfit values on lune
    """
    gamma, delta = np.meshgrid(gamma, delta)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = values.flatten()

    minval = values.min()
    maxval = values.max()

    if minval==maxval:
        warnings.warn(
            "Nothing to plot: all values are identical",
            Warning)
        return

    if maxval-minval < 1.e-6:
        exp = -np.fix(np.log10(maxval-minval))
        warnings.warn(
           "Multiplying by 10^%d to avoid GMT plotting errors" % exp,
           Warning)
        values *= 10.**exp
        minval *= 10.**exp
        maxval *= 10.**exp


    #
    # prepare gmt input
    #

    zmin_zmax_dz = '%e/%e/%e' % (minval, maxval, (maxval-minval)/100.)
    title = _parse(title)
    name, fmt = check_ext(filename)

    # FIXME: can GMT accept virtual files?
    tmpname = 'tmp_'+name+'.txt'
    np.savetxt(tmpname, np.column_stack([gamma, delta, values]))


    #
    # call gmt script
    #

    if gmt_cmd():
        _call("%s %s %s %s %s %s" %
           (fullpath('mtuq/graphics/_gmt/plot_lune'),
            tmpname,
            filename,
            fmt,
            zmin_zmax_dz,
            title
            ))
    else:
        gmt_not_found_warning(
            tmpname)


def gmt_not_found_warning(filename):
    warnings.warn("""
        WARNING

        Generic Mapping Tools executables not found on system path.
        PostScript output has not been written. 

        Misfit values have been saved to:
            %s
        """ % filename)


def _call(cmd):
    subprocess.call(cmd, shell=True)


def _parse(title):
    if not title:
        return ""

    title_args = ''
    for part in title.split("\n"):
        title_args += "'"+part+"' "
    return title_args



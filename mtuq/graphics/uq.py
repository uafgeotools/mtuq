
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
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import fullpath
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.xarray import dataarray_to_table


def plot_misfit(filename, struct, title=None):
    """ Plots misfit values on eigenvalue lune (requires GMT)


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``struct`` (`DataArray` or `DataFrame`):
    Structure containing moment tensors and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given as a
    `DataArray` and `DataFrame`.

    `DataArrays` and `DataFrames` can be used to represent regularly-spaced
    and irregularly-spaced grids, respectively.  These structures make
    multidimensional `min`, `max` and `sum` operations easy, so they are used
    here for projecting from 6-D moment tensor space onto 2-D lune space.

    For converting to `DataArrays` and `DataFrames` from MTUQ grid types, see
    `mtuq.grid.Grid.to_datarray` and
    `mtuq.grid.UnstructuredGrid.to_dataframe`.


    .. note ::

      This utility requires Generic Mapping Tools >=5.

      To display information about supported image formats, type
      `gmt psconvert --help1.

      For a matplotlib-only alternative, see `plot_misfit_vw`.

    """
    struct = struct.copy()


    if type(struct)==DataArray:
        da = struct
        da = da.min(dim=('rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        values = da.values


    elif type(struct)==DataFrame:
        df = struct
        gamma, delta, values = _bin(df, lambda df: df.min())


    _plot_lune(filename, gamma, delta, values, title)



def plot_likelihood(filename, struct, sigma=1., title=None):
    """ Plots maximum likelihoods on eigenvalue lune (requires GMT)


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``struct`` (`DataArray` or `DataFrame`):
    Structure containing moment tensors and corresponding misfit values

    ``sigma`` (`float`):
    Standard deviation applied to misfit values to obtain likelihood values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given as a
    `DataArray` and `DataFrame`.

    `DataArrays` and `DataFrames` can be used to represent regularly-spaced
    and irregularly-spaced grids, respectively.  These structures make
    multidimensional `min`, `max` and `sum` operations easy, so they are used
    here for projecting from 6-D moment tensor space onto 2-D lune space.

    For converting to `DataArrays` and `DataFrames` from MTUQ grid types, see
    `mtuq.grid.Grid.to_datarray` and
    `mtuq.grid.UnstructuredGrid.to_dataframe`.


    .. note ::

      This utility requires Generic Mapping Tools >=5.

      To display information about supported image formats, type
      `gmt psconvert --help1.

      For a matplotlib-only alternative, see `plot_misfit_vw`.

    """
    struct = struct.copy()


    # convert from misfit to likelihood
    struct.values = np.exp(-struct.values/(2.*sigma**2))


    if type(struct)==DataArray:
        da = struct
        da = da.max(dim=('rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        values = da.values


    elif type(struct)==DataFrame:
        df = struct
        gamma, delta, values = _bin(df, lambda df: df.max())


    values /= values.sum()

    _plot_lune(filename, gamma, delta, values, title)



def plot_marginal(filename, struct, sigma=1., title=None):
    """ Plots marginal likelihoods on eigenvalue lune (requires GMT)
    
    
    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``struct`` (`DataArray` or `DataFrame`):
    Structure containing moment tensors and corresponding misfit values

    ``sigma`` (`float`):
    Standard deviation applied to misfit values to obtain likelihood values
        
    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Moment tensors and corresponding misfit values must be given as a
    `DataArray` and `DataFrame`.

    `DataArrays` and `DataFrames` can be used to represent regularly-spaced
    and irregularly-spaced grids, respectively.  These structures make
    multidimensional `min`, `max` and `sum` operations easy, so they are used
    here for projecting from 6-D moment tensor space onto 2-D lune space.

    For converting to `DataArrays` and `DataFrames` from MTUQ grid types, see
    `mtuq.grid.Grid.to_datarray` and
    `mtuq.grid.UnstructuredGrid.to_dataframe`.

        
    .. note ::

      This utility requires Generic Mapping Tools >=5.

      To display information about supported image formats, type
      `gmt psconvert --help1.

      For a matplotlib-only alternative, see `plot_misfit_vw`.
 
    """

    struct = struct.copy()


    # convert from misfit to likelihood
    struct.values = np.exp(-struct.values/(2.*sigma**2))


    if type(struct)==DataArray:
        da = struct
        da = da.sum(dim=('rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        values = da.values


    elif type(struct)==DataFrame:
        df = struct
        gamma, delta, values = _bin(df, lambda df: df.sum()/len(df))

    values /= lune_det(delta, gamma)
    values /= values.sum()

    _plot_lune(filename, gamma, delta, values)



#
# utilities for irregularly-spaced grids
#

def _bin(df, handle, npts_v=40, npts_w=20, tightness=0.8):
    """ Bins DataFrame into rectangular cells
    """
    # at which points will we plot values?
    centers_v, centers_w = semiregular_grid(
        npts_v, npts_w, tightness=tightness)

    centers_gamma, centers_delta = to_gamma(centers_v), to_delta(centers_w)


    # what cell edges correspond to the above centers?
    edges_v = np.array(centers_v[:-1] + centers_v[1:])/2.
    edges_v = np.pad(edges_v, 2)
    edges_v[0] = -1./3.; edges_v[-1] = +1./3.

    edges_w = np.array(centers_w[:-1] + centers_w[1:])/2.
    edges_w = np.pad(edges_w, 2)
    edges_w[0] = -3.*np.pi/8.; edges_w[-1] = +3.*np.pi/8

    edges_gamma, edges_delta = to_gamma(edges_v), to_delta(edges_w)


    # bin grid points into cells
    binned = np.empty((npts_delta, npts_gamma))
    for _i in range(npts_delta):
        for _j in range(npts_gamma):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['gamma'].between(edges_gamma[_j], edges_gamma[_j+1]) &
                df['delta'].between(edges_delta[_i], edges_delta[_i+1])]

            binned[_i, _j] = handle(subset['values'])

            # normalize by area of cell
            binned[_i, _j] /= edges_v[_j+1] - edges_v[_j]
            binned[_i, _j] /= edges_w[_i+1] - edges_w[_i]


    return centers_gamma, centers_delta, binned



#
# GMT wrappers
#

def _plot_lune(filename, gamma, delta, values, title=None):
    """ Plots misfit values on lune
    """
    delta, gamma = np.meshgrid(delta, gamma)
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



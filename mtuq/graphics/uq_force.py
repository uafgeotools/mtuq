
#
# graphics/uq_force.py - uncertainty quantification of forces on the unit sphere
#

import numpy as np
import subprocess
import warnings

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray

from mtuq.graphics._gmt import gmt_cmd, gmt_not_found_warning, check_ext
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import fullpath
from mtuq.util.math import closed_interval, open_interval


def plot_misfit_force(filename, ds, title=None):
    """ Plots misfit values on `v-w` rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing forces and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Forces and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)

    """
    _check(ds)
    ds = ds.copy()


    if issubclass(type(ds), DataArray):
        da = ds.min(dim=('origin_idx', 'F0'))
        phi = da.coords['phi']
        h = da.coords['h']
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        phi, h, values = _bin(df, lambda df: df.min())


    _plot_force_gmt(filename, phi, h, values, title=title)



def plot_likelihood_force(filename, ds, sigma=1., title=None):
    """ Plots maximum likelihoods on `v-w` rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing forces and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Forces and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)

    """
    _check(ds)
    ds = ds.copy()


    # convert from misfit to likelihood
    ds.values = np.exp(-ds.values/(2.*sigma**2))
    ds.values /= ds.values.sum()



    if issubclass(type(ds), DataArray):
        da = ds.max(dim=('origin_idx', 'F0'))
        phi = da.coords['phi']
        h = da.coords['h']
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        df['values'] /= df['values'].sum()
        phi, h, values = _bin(df, lambda df: df.max())


    values /= values.sum()
    values /= vw_area

    _plot_force_gmt(filename, phi, h, values, title=title)



def plot_marginal_force(filename, ds, sigma=1., title=None):
    """ Plots marginal likelihoods on `v-w` rectangle


    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    data structure containing forces and corresponding misfit values

    ``title`` (`str`):
    Optional figure title


    .. rubric :: Usage

    Forces and corresponding misfit values must be given in the format
    returned by `mtuq.grid_search` (in other words, as a `DataArray` or 
    `DataFrame`.)


    """
    _check(ds)
    ds = ds.copy()


    # convert from misfit to likelihood
    ds.values = np.exp(-ds.values/(2.*sigma**2))
    ds.values /= ds.values.sum()


    if issubclass(type(ds), DataArray):
        da = ds.sum(dim=('origin_idx', 'F0'))
        phi = da.coords['phi']
        h = da.coords['h']
        values = da.values.transpose()


    elif issubclass(type(ds), DataFrame):
        df = ds.reset_index()
        phi, h, values = _bin(df, lambda df: df.sum()/len(df))


    values /= values.sum()
    values /= vw_area

    _plot_force_gmt(filename, phi, h, values, title=title)



def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")



#
# utilities for irregularly-spaced grids
#


def _bin(df, handle, npts_phi=60, npts_h=30):
    """ Bins DataFrame into rectangular cells
    """
    # define centers of cells
    centers_phi = open_interval(0., 360., npts_phi)
    centers_h = open_interval(-1., +1., npts_h)

    # define corners of cells
    phi = closed_interval(0., 360, npts_phi+1)
    h = closed_interval(-1., +1., npts_h+1)

    binned = np.empty((npts_h, npts_phi))
    for _i in range(npts_h):
        for _j in range(npts_phi):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['phi'].between(phi[_j], phi[_j+1]) &
                df['h'].between(h[_i], h[_i+1])]

            if len(subset)==0:
                print("Encountered empty bin\n"
                      "phi: %f, %f\n"
                      "h: %f, %f\n" %
                      (phi[_j], phi[_j+1], h[_i], h[_i+1]) )

            binned[_i, _j] = handle(subset[0])

    return centers_phi, centers_h, binned



#
# pyplot wrappers
#

def _plot_force(filename, phi, h, values):
    """ Plots misfit values on sphere (matplotlib implementation)
    """ 
    raise NotImplementedError



#
# GMT wrappers
#


def _wrap(angle_in_deg):
    """ Wraps angle to (-180, 180)
    """
    angle_in_deg %= 360.
    idx = np.where(angle_in_deg > 180.)
    angle_in_deg[idx] -= 360.
    return angle_in_deg


def _plot_force_gmt(filename, phi, h, values, add_marker=True, title=''):
    """ Plots misfit values on sphere (GMT implementation)
    """
    lat = np.degrees(np.pi/2 - np.arccos(h))
    lon = _wrap(phi + 90.)

    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()
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

    name, fmt = check_ext(filename)

    zmin_zmax_dz = '%e/%e/%e' % (minval, maxval, (maxval-minval)/100.)

    if add_marker:
        idx = values.argmin()
        marker_coords = "'%f %f'" % (lon[idx], lat[idx])
    else:
        marker_coords = "''"

    try:
        parts = title.split('\n')
    except:
        parts = []

    if len(parts) >= 2:
        title = "'%s'" % parts[0]
        subtitle = "'%s'" % parts[1]
    elif len(parts) == 1:
        title = "'%s'" % parts[0]
        subtitle = "''"
    else:
        title = "''"
        subtitle = "''"


    # FIXME: can GMT accept virtual files?
    tmpname = 'tmp_'+name+'.txt'
    np.savetxt(tmpname, np.column_stack([lon, lat, values]))


    #
    # call gmt script
    #

    if gmt_cmd():
        _call("%s %s %s %s %s %s %s %s" %
           (fullpath('mtuq/graphics/_gmt/plot_force'),
            tmpname,
            filename,
            fmt,
            zmin_zmax_dz,
            marker_coords,
            title,
            subtitle
            ))

    else:
        gmt_not_found_warning(
            tmpname)



def _call(cmd):
    subprocess.call(cmd, shell=True)



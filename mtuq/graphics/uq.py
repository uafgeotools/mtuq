
#
# graphics/uq.py - uncertainty quantification on the eigenvalue lune
#

import numpy as np
import shutil
import subprocess

from matplotlib import pyplot
from os.path import splitext
from pandas import DataFrame
from xarray import DataArray
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.util import fullpath
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.xarray import dataarray_to_table


def plot_misfit(filename, struct):
    """ Plots misfit on eigenvalue lune
    """
    struct = struct.copy()
    struct.values -= struct.values.min()
    struct.values /= struct.values.max()


    if type(struct)==DataArray:
        da = struct.copy()
        da = da.min(dim=('rho', 'kappa', 'sigma', 'h'))
        gamma = to_gamma(da.coords['v'])
        delta = to_delta(da.coords['w'])
        _plot_lune(filename, gamma, delta, da.values)


    elif type(struct)==DataFrame:
        df = struct.copy()
        gamma, delta, values = _bin(df, lambda df: df.min())
        _plot_lune(filename, gamma, delta, da.values)


def _plot_lune(filename, gamma, delta, values):
    """ Plots misfit values on lune
    """
    delta, gamma = np.meshgrid(delta, gamma)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = values.flatten() 

    # write misfit values
    name, ext = _check_ext(filename)
    tmpname = 'tmp_'+name+'.txt'
    np.savetxt(tmpname, np.column_stack([gamma, delta, values]))

    # write PostScript graphics
    if _gmt():
        _call("%s %s %s" %
           (fullpath('mtuq/graphics/_gmt/plot_misfit'),
            tmpname,
            name+ext))
    else:
        gmt_not_found_warning(
            tmpname)


def _bin(df, handle, npts_delta=40, npts_gamma=20, tightness=0.8):
    """ Bins DataFrame into rectangular cells
    """
    npts_v, npts_w = npts_gamma, npts_delta
    v, w = semiregular_grid(npts_v, npts_w)

    centers_gamma = to_gamma(v)
    centers_delta = to_delta(w)

    # what cell edges correspond to the above cell centers?
    gamma = np.array(centers_gamma[:-1] + centers_gamma[1:])/2.
    gamma = np.pad(gamma, 2)
    gamma[0] = -30.; gamma[-1] = +30.
    delta = np.array(centers_delta[:-1] + centers_delta[1:])/2.
    delta = np.pad(delta, 2)
    delta[0] = -90.; delta[-1] = +90.

    binned = np.empty((npts_delta, npts_gamma))
    for _i in range(npts_delta):
        for _j in range(npts_gamma):
            # which grid points lie within cell (i,j)?
            subset = df.loc[
                df['gamma'].between(gamma[_j], gamma[_j+1]) &
                df['delta'].between(delta[_i], delta[_i+1])]

            binned[_i, _j] = handle(subset['values'])

    return centers_gamma, centers_delta, binned


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


def _gmt():
    return shutil.which('gmt')


def _check_ext(filename):
    name, ext = splitext(filename)

    if ext.lower()!='ps':
        print('Appending extension ".ps" to PostScript file')
        return name, '.ps'
    else:
        return name, '.'+ext


def _centers_to_edges(v):
    raise NotImplementedError

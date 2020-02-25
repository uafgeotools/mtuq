
import numpy as np
import shutil
import subprocess

from matplotlib import pyplot
from os.path import splitext
from xarray import DataArray
from mtuq.graphics.uq import check_grid
from mtuq.util import fullpath
from mtuq.util.lune import to_delta, to_gamma
from mtuq.util.xarray import dataarray_to_table



def plot_misfit(filename, grid, misfit):
    """ Plots misfit values on lune
    (GMT implementation)
    """
    da = check_grid('FullMomentTensor', grid)
    
    # manipulate DataArray
    da = da.min(['rho', 'kappa', 'sigma', 'h'])
    da.values -= da.values.min()
    da.values /= da.values.max()

    # get coordinates
    delta = to_delta(da.coords['w'])
    gamma = to_gamma(da.coords['v'])
    delta, gamma = np.meshgrid(delta, gamma)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = da.values.flatten() 

    # write misfit values
    name, ext = _check_ext(filename)
    tmpname = 'tmp_'+name+'_likelihood.txt'
    np.savetxt(tmpname, np.column_stack([gamma, delta, values]))

    # write PostScript graphics
    if _gmt():
        _call("%s %s %s" %
           (fullpath('scripts/plot_misfit'),
            tmpname,
            name+ext))
    else:
        gmt_not_found_warning(
            tmpname)


def plot_likelihood(filename, grid, misfit):
    """ Plots likelihood values on lune
    (GMT implementation)
    """
    da = check_grid('FullMomentTensor', grid)

    # manipulate DataArray
    da.values = np.exp(-da.values)
    da = da.sum(['rho', 'kappa', 'sigma', 'h'])
    da.values -= da.values.min()
    da.values /= da.values.max()

    # get coordinates
    delta = to_delta(da.coords['w'])
    gamma = to_gamma(da.coords['v'])
    delta, gamma = np.meshgrid(delta, gamma)
    delta = delta.flatten()
    gamma = gamma.flatten()
    values = da.values.flatten()

    # write misfit values
    name, ext = _check_ext(filename)
    tmpname = 'tmp_'+name+'.txt'
    np.savetxt(tmpname, np.column_stack([gamma, delta, values]))

    # write PostScript graphics
    if _gmt():
        _call("%s %s %s" %
           (fullpath('scripts/plot_misfit'),
            tmpname,
            name+ext))
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


def _gmt():
    return shutil.which('gmt')


def _check_ext(filename):
    name, ext = splitext(filename)

    if ext.lower()!='ps':
        print('Appending extension ".ps" to PostScript file')
        return name, '.ps'
    else:
        return name, '.'+ext



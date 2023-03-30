
# 
# graphics/uq/omega.py - uncertainty quantification over force or moment tensor
# angular distance
#

import numpy as np

from mtuq import Force, MomentTensor
from mtuq.graphics.uq._matplotlib import _plot_omega_matplotlib
from mtuq.grid_search import DataArray, DataFrame
from mtuq.util import warn
from mtuq.util.math import to_mij
from pandas import DataFrame

ISO = MomentTensor([1.,1.,1.,1.,1.,1.])



def plot_pdf(filename, df, var, m0=None, nbins=50, **kwargs):
    """ Plots probability density function over angular distance

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``df`` (`DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance

    """
    if not isuniform(df):
        warn('plot_pdf requires randomly-drawn grid')
        return

    omega, pdf = _calculate_pdf(df, var, m0=m0, nbins=nbins)
    _plot_omega(filename, omega, pdf, **kwargs)



def plot_cdf(filename, df, var, nbins=50, **kwargs):
    """ Plots cumulative distribution function over angular distance

    .. rubric :: Input arguments

    ``filename`` (`str`):
    Name of output image file

    ``df`` (`DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

    ``var`` (`float` or `array`):
    Data variance

    """
    if not isuniform(df):
        warn('plot_cdf requires randomly-drawn grid')
        return

    omega, pdf = _calculate_pdf(df, var, m0=m0, nbins=nbins)
    _plot_omega(filename, omega, np.cumsum(pdf), **kwargs)



def plot_screening_curve(filename, ds, var, nbins=50,
    backend=_plot_omega_matplotlib, **kwargs):
    """ Plots explosion screening curve

    In other words, plots probability density function that results from 
    integrating outward in angular distance from an isotropic moment tensor
    """
    if issubclass(type(ds), DataArray):
        omega, values = _screening_curve_regular(ds, var, nbins=nbins)

    elif issubclass(type(ds), DataFrame):
        omega, values = _screening_curve_random(ds, var, nbins=nbins)

    _plot_omega(filename, omega, values, **kwargs)



def _plot_omega(filename, omega, values, backend=_plot_omega_matplotlib):
    backend(filename, omega, values)



def _calculate_pdf(df, var, m0=None, nbins=50, normalize=False):
    """ Calculates probability density function over angular distance
    """

    # convert DataFrame to NumPy array
    if _type(df)=='MomentTensor':
        array = _to_array(df)

    elif _type(df)=='Force':
        raise NotImplementedError

    else:
        raise TypeError

    if type(m0)==MomentTensor:
        # convert from lune to GCMT parameters
        m0 = m0.as_vector()

    elif type(m0)==Force:
        raise NotImplementedError

    elif not m0:
        # use maximum likelihood estimate
        idx = _argmax(df)
        m0 = array[idx,:]


    #
    # calculate likelihoods and corresponding angular distances
    #

    df = df.copy()
    df = np.exp(-df/(2.*var))
    samples = df[0].values

    # vectorized dot product
    omega = np.dot(array, m0)
    omega /= np.sum(m0**2)**0.5
    omega /= np.sum(array**2, axis=1)**0.5

    omega = np.arccos(omega)
    omega *= 180./np.pi


    #
    # Monte Carlo integration
    #

    centers = np.linspace(0, 180., nbins+2)[1:-1]
    edges = np.linspace(0., 180., nbins+1)
    delta = np.diff(edges)

    likelihoods = np.zeros(nbins)
    counts = np.zeros(nbins)

    # bin likelihoods by angular distance
    for _i in range(nbins):
        indices = np.argwhere(np.abs(omega - centers[_i]) < delta[_i])

        if indices.size==0:
            print('Encountered empty bin')
            continue

        for index in indices:
            likelihoods[_i] += samples[index]
            counts[_i] += 1

    if normalize:
        mask = np.argwhere(counts > 0)
        likelihoods[mask] /= counts[mask]
        centers = centers[mask]
        likelihoods = likelihoods[mask]

    likelihoods /= sum(likelihoods)
    likelihoods /= 180.

    return centers, likelihoods


def _screening_curve_random(df, var, nbins=50):
    return _calculate_pdf(df, var, m0=ISO, nbins=nbins, normalize=True)


def _screening_curve_regular(da, var):
    raise NotImplementedError


#
# utility functions
#

def _argmax(df):

    df = df.copy()
    df = df.reset_index()
    return df[0].idxmax()


def _to_array(df):

    df = df.copy()
    df = df.reset_index()

    try:
        return np.ascontiguousarray(to_mij(
            df['rho'].to_numpy(),
            df['v'].to_numpy(),
            df['w'].to_numpy(),
            df['kappa'].to_numpy(),
            df['sigma'].to_numpy(),
            df['h'].to_numpy(),
            ))
    except:
        raise TypeError


def _type(ds):

    try:
        # DataArray
        dims = ds.dims
    except:
        # DataFrame
        ds = ds.copy()
        ds = ds.reset_index()
        dims = list(ds.columns.values)

    if 'rho' in dims\
       and 'v' in dims\
       and 'w' in dims\
       and 'kappa' in dims\
       and 'sigma' in dims\
       and 'h' in dims:
        return 'MomentTensor'

    elif 'F0' in dims\
       and 'phi' in dims\
       and 'h' in dims:
        return 'Force'


def isuniform(ds):
    if issubclass(type(ds), DataFrame):
        return True



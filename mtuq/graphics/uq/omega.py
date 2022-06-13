
# 
# graphics/uq/omega.py - uncertainty quantification over moment tensor angular distance
#

import numpy as np

from mtuq.graphics.uq._matplotlib import _plot_omega_matplotlib
from mtuq.util import warn
from mtuq.util.math import to_mij
from pandas import DataFrame



def plot_pdf(filename, df, var, nbins=50, backend=_plot_omega_matplotlib, **kwargs):
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

    omega, pdf = _calculate_pdf(df, var, nbins)
    backend(filename, omega, pdf, **kwargs)



def plot_cdf(filename, df, var, nbins=50, backend=_plot_omega_matplotlib, **kwargs):
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

    omega, pdf = _calculate_pdf(df, var, nbins)
    backend(filename, omega, np.cumsum(pdf), **kwargs)



def plot_screening_curve(filename, ds, var, **kwargs):
    """ Plots explosion screening curve

    In other words, plots probability density function that results from 
    integrating outward in angular distance from an isotropic moment tensor
    """

    if issubclass(type(ds), DataArray):
        omega, values = _screening_curve_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        omega, values = _screening_curve_random(ds, var)

    backend(filename, omega, values, **kwargs)



def _calculate_pdf(df, var, nbins):
    """ Calculates probability density function over angular distance
    """

    # convert misfit to likelihood
    df = df.copy()
    df = np.exp(-df/(2.*var))

    # convert from lune to Cartesian parameters
    mt_array = _to_array(df)

    # maximum likelihood estimate
    idx = _argmax(df)
    mt_best = mt_array[idx,:]

    # structure containing randomly-sampled likelihoods
    samples = df[0].values


    #
    # calculate angular distance
    #

    # vectorized dot product
    omega_array = np.dot(mt_array, mt_best)
    omega_array /= np.sum(mt_best**2)**0.5
    omega_array /= np.sum(mt_array**2, axis=1)**0.5

    # calculate angles in degrees
    omega_array = np.arccos(omega_array)
    omega_array *= 180./np.pi


    #
    # Monte Carlo integration
    #

    centers = np.linspace(0, 180., nbins+2)[1:-1]
    edges = np.linspace(0., 180., nbins+1)
    delta = np.diff(edges)

    # bin likelihoods by angular distance
    likelihoods = np.zeros(nbins)
    for _i in range(nbins):
        indices = np.argwhere(np.abs(omega_array - centers[_i]) < delta[_i])

        if indices.size==0:
            print('Encountered empty bin')
            continue

        for index in indices:
            likelihoods[_i] += samples[index]

    likelihoods /= sum(likelihoods)
    likelihoods /= 180.

    return centers, likelihoods


def _screening_curve_random(df, var):
    raise NotImplementedError

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


def isuniform(ds):
    if issubclass(type(ds), DataFrame):
        return True



# 
# graphics/uq/omega.py - uncertainty quantification over moment tensor angular distance
#

import numpy as np

from mtuq.graphics.uq._matplotlib import _plot_omega_matplotlib, _plot_omega_matplotlib_test, _plot_rho_av
from mtuq.util import warn
from mtuq.util.math import to_mij
from pandas import DataFrame
from scipy.interpolate import interp1d



def plot_pdf(filename, df, var, nbins, backend=_plot_omega_matplotlib, **kwargs):
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
    omega, pdf, likelihoods_homo =_calculate_pdf(df, var, nbins=nbins)
    backend(filename, omega, pdf, likelihoods_homo, **kwargs)

    return omega



def plot_cdf(filename, df, var, nbins, backend=_plot_omega_matplotlib, **kwargs):
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

    omega, likelihoods, likelihoods_homo =_calculate_pdf(df, var,nbins=nbins)
    backend(filename, omega, np.cumsum(likelihoods), np.cumsum(likelihoods_homo), **kwargs)



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
    
def _screening_curve_random(df, var):
    raise NotImplementedError

def _screening_curve_regular(da, var):
    raise NotImplementedError
    
def misfit_vs_omega(filename,df,backend=_plot_omega_matplotlib_test):
    """ Plots the misfit values with corresponding angular distance
    """
     
    mt_array = _to_array(df)
    idx = _argMaxMin(df, False)
    mt_best = mt_array[idx,:]
    omega_array = _compute_omega(mt_array,mt_best)
     
     
    backend(filename,omega_array,df.values)
    
def probability_vs_omega(filename, df,backend=_plot_omega_matplotlib_test):
     """ Plots the probability values with corresponding angular distance
    """
     mt_array = _to_array(df)
     idx = _argMaxMin(df, False)
     mt_best = mt_array[idx,:]
     omega_array = _compute_omega(mt_array,mt_best)
     prob = np.exp(-(df.values))


     backend(filename, omega_array, prob)
     
def _compute_omega(mt1,mt2):
     """ evaluates the angular between referece MT and other MTs. The reference 
     one is the MT with minimum misfit"""
     omega_array = np.dot(mt1, mt2)
     omega_array /= np.sum(mt2**2)**0.5
     omega_array /= np.sum(mt1**2, axis=1)**0.5
    
     omega_array = np.arccos(omega_array)
     omega_array *= 180./np.pi    
     return omega_array



def _calculate_pdf(df, var, nbins):
    """ Calculates probability density function over angular distance
    """
    df = df.copy()
    df = np.exp(-df*(10**10))*var
    df_homo = np.exp(-df*0*(10**10))*var


    # convert from lune to Cartesian parameters
    mt_array = _to_array(df)
    mt_array_homo = _to_array(df_homo)
    # maximum likelihood estimate

    idx =  _argMaxMin(df,True)
    idx_homo = _argMaxMin(df_homo,True)
    mt_best = mt_array[idx,:]
    mt_best_homo = mt_array_homo[idx_homo,:]


    # structure containing randomly-sampled likelihoods
    samples = df[0].values
    samples_homo = df_homo[0].values

    #
    # calculate angular distance
    # vectorized dot product
    omega_array = np.dot(mt_array, mt_best)
    omega_array_homo = np.dot(mt_array_homo, mt_best_homo)


    omega_array /= np.sum(mt_best**2)**0.5
    omega_array /= np.sum(mt_array**2, axis=1)**0.5

    omega_array_homo /= np.sum(mt_best_homo**2)**0.5
    omega_array_homo /= np.sum(mt_array_homo**2, axis=1)**0.5
 
    # calculate angles in degrees
    omega_array = np.arccos(omega_array)
    omega_array *= 180./np.pi


    omega_array_homo = np.arccos(omega_array_homo)
    omega_array_homo *= 180./np.pi


    #-------------------------------------------------------------
    # Monte Carlo integration
    #-------------------------------------------------------------
    centers = np.linspace(0, 180., nbins+3)[1:-1]
    edges = np.linspace(0., 180., nbins+1)
    delta = np.diff(edges)

    # bin likelihoods by angular distance
    likelihoods = np.zeros(nbins)
    likelihoods_homo = np.zeros(nbins)

    my_list = []
    for _i in range(nbins):
        indices = np.argwhere(np.abs(omega_array - centers[_i]) < delta[_i])
        indices_homo = np.argwhere(np.abs(omega_array_homo - centers[_i]) < delta[_i])

        if indices.size==0:
            print('Encountered empty bin')
            continue
        for index in indices:
            likelihoods[_i] += samples[index]

        if indices_homo.size==0:
            print('Encountered empty bin')
            continue
        for index_homo in indices_homo:
            likelihoods_homo[_i] += samples_homo[index_homo]
    likelihoods /= sum(likelihoods)
    likelihoods_homo /= sum(likelihoods_homo)

    likelihoods = np.append(0, likelihoods)
    likelihoods_homo = np.append(0, likelihoods_homo)
    

    return centers, likelihoods, likelihoods_homo
    
def plot_rho_vs_V(filename, df, unc, var, nbins, backend=_plot_rho_av, **kwargs):
    v, rho_v, rho_v_homo = _calculate_pdf(df, var,nbins=nbins)
    rho_v = np.cumsum(rho_v)
    rho_v_homo = np.cumsum(rho_v_homo)
    for i in v:
        a = interp1d(v, rho_v_homo)(v)
        b = interp1d(v, rho_v)(v)

    backend(filename, a, b, unc, **kwargs)


#
# utility functions
#

def _argMaxMin(df, idMinMax: bool):
    df = df.copy()
    df = df.reset_index()
    if idMinMax:
        return df[0].idxmax()
    return df[0].idxmin()


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


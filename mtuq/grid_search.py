
import numpy as np
from mtuq.util.grid import Grid, UnstructuredGrid
from mtuq.util.math import PI
from mtuq.util.util import asarray, timer, timer_mpi

from mtuq.util.moment_tensor import tape2015
from numpy.random import uniform as random
from mtuq.util.math import open_interval as regular



@timer
def grid_search_serial(data, greens, misfit, grid):
    """ 
    Grid search over moment tensors
    """
    results = np.zeros(grid.size)
    count = 0

    for mt in grid:
        print grid.index

        # generate_synthetics
        synthetics = {}
        for key in data:
            synthetics[key] = greens[key].get_synthetics(mt)

        # evaluate misfit
        for key in data:
            func, dat, syn = misfit[key], data[key], synthetics[key]
            results[count] += func(dat, syn)
        count += 1

    return results


@timer_mpi
def grid_search_mpi(data, greens, misfit, grid):
    """
    To carry out a grid search in parallel, we decompose the grid into subsets 
    and scatter using MPI. Each MPI process then runs grid_search_serial on its
    assigned subset
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    if iproc == 0:
        subset = grid.decompose(nproc)
    else: 
        subset = None
    subset = comm.scatter(subset, root=0)

    return grid_search_serial(data, greens, misfit, subset)



def MTGridRandom(Mw=[], npts=50000):
    """ Full moment tensor grid with randomly-spaced values
    """
    N = npts

    # upper bound, lower bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = float(Mw)/np.sqrt(2)

    return UnstructuredGrid({
        'rho': rho*np.ones(N),
        'v': random(*v),
        'w': random(*w),
        'kappa': random(*kappa),
        'sigma': random(*sigma),
        'h': random(*h)},
        callback=tape2015.tt152cmt)


def MTGridRegular(Mw, npts_per_axis=25):
    """ Full moment tensor grid with regularly-spaced values
    """
    N = npts_per_axis

    # upper bound, lower bound, number of points
    v = [-1./3., 1./3., N]
    w = [-3./8.*PI, 3./8.*PI, N]
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = asarray(Mw)/np.sqrt(2)

    return Grid({
        'rho': rho,
        'v': regular(*v),
        'w': regular(*w),
        'kappa': regular(*kappa),
        'sigma': regular(*sigma),
        'h': regular(*h)},
        callback=tape2015.tt152cmt)


def DCGridRandom(Mw, npts=50000):
    """ Double-couple moment tensor grid with randomly-spaced values
    """
    N = npts

    # upper bound, lower bound, number of points
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = float(Mw)/np.sqrt(2)

    return UnstructuredGrid({
        'rho': rho*np.ones(N),
        'v': np.zeros(N),
        'w': np.zeros(N),
        'kappa': random(*kappa),
        'sigma': random(*sigma),
        'h': random(*h)},
        callback=tape2015.tt152cmt)


def DCGridRegular(Mw, npts_per_axis=25):
    """ Double-couple moment tensor grid with regularly-spaced values
    """ 
    N = npts_per_axis

    # upper bound, lower bound, number of points
    kappa = [0., 360, N]
    sigma = [-90., 90., N]
    h = [0., 1., N]

    # magnitude is treated separately
    rho = asarray(Mw)/np.sqrt(2)

    return Grid({
        'rho': rho,
        'v': np.array([0.]),
        'w': np.array([0.]),
        'kappa': regular(*kappa),
        'sigma': regular(*sigma),
        'h': regular(*h)},
        callback=tape2015.tt152cmt)


def OriginGrid():
    raise NotImplementedError


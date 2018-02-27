
import time
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.util import is_mpi_env, Struct


def grid_search_serial(data, greens, misfit, grid):
    """ Grid search over moment tensor parameters
    """
    misfit_values = np.zeros(grid.size)

    for _i in range(grid.size):
        print _i

        # gets the i-th moment tensor in grid
        mt = grid.get(_i)

        # generate_synthetics
        synthetics = {}
        for key in data:
            synthetics[key] = greens[key].get_synthetics(mt)

       # evaluate misfit
        for key in data:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            misfit_values[_i] += chi(dat, syn)


def grid_search_mpi(data, greens, misfit, grid):
    from mpi4py import MPI
    from schwimmbad import MPIPool
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    # DEBUG
    start_time = time.time()

    with MPIPool() as pool:
        # constuct arguments list
        tasks = []
        for subset in grid.decompose(nproc-1):
            tasks += [[data, greens, misfit, subset]]

        # evaluate misfit
        results = pool.map(
            _evalutate_misfit,
            tasks)

        # DEBUG
        elapsed_time = time.time() - start_time
        if iproc==0:
            print 'ELAPSED TIME:', elapsed_time


def _evalutate_misfit(args):
    data, greens, misfit, grid = args

    results = np.zeros(grid.size)
    for _i in range(grid.size):
        print _i

        # get i-th moment tensor in grid
        mt = grid.get(_i)

        # generate synthetics
        synthetics = {}
        for key in data:
            synthetics[key] = greens[key].get_synthetics(mt)

        # sum data categories
        _sum = 0.
        for key in data:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            _sum += chi(dat, syn)
        results[_i] = _sum

    return results



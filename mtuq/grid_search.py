
import time
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.util import is_mpi_env, Struct

# adds debugging output to grid_search_mpi
PRINT_ELAPSED_TIME=True



def grid_search_serial(data, greens, misfit, grid):
    """ Grid search over moment tensor parameters
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
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            results[count] += chi(dat, syn)

        count += 1


def grid_search_mpi(data, greens, misfit, grid):
    from mpi4py import MPI
    from schwimmbad import MPIPool
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    with MPIPool() as pool:
        # constuct arguments list
        tasks = []
        for subset in grid.decompose(nproc-1):
            tasks += [[data, greens, misfit, subset]]

        # evaluate misfit
        results = pool.map(
            _evalutate_misfit,
            tasks)


def _evalutate_misfit(args):
    data, greens, misfit, grid = args

    # array to hold misfit values
    results = np.zeros(grid.stop-grid.start)
    count = 0

    for mt in grid:
        print grid.index

        # generate synthetics
        synthetics = {}
        for key in data:
            synthetics[key] = greens[key].get_synthetics(mt)

        # sum over data categories
        for key in data:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            results[count] += chi(dat, syn)

        count += 1

    return results


if PRINT_ELAPSED_TIME:
    # adds debugging output to grid_search_mpi
    _old = grid_search_mpi

    def grid_search_mpi(data, greens, misfit, grid):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        if comm.rank==0:
            start_time = time.time()

        _old(data, greens, misfit, grid)

        if comm.rank==0:
            elapsed_time = time.time() - start_time
            print 'Elapsed time:', elapsed_time


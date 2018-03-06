
import numpy as np

from util.util import elapsed_time, elapsed_time_mpi



@elapsed_time
def grid_search_serial(data, greens, misfit, grid):
    """ Grid search over moment tensor parameters
    """
    results = np.zeros(grid.size)
    count = 0

    for mt in grid:
        #print grid.index

        # generate_synthetics
        synthetics = {}
        for key in data:
            synthetics[key] = greens[key].get_synthetics(mt)

       # evaluate misfit
        for key in data:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            results[count] += chi(dat, syn)

        count += 1


@elapsed_time_mpi
def grid_search_mpipool(data, greens, misfit, grid):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    with MPIPool() as pool:
        # constuct arguments list
        tasks = []
        for subset in grid.decompose(nproc-1):
            tasks += [[data, greens, misfit, subset]]

        # evaluate misfit
        results = pool.map(
            _evaluate_misfit,
            tasks)


@elapsed_time_mpi
def grid_search_mpi(data, greens, misfit, grid):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    # scatter grid across processes
    if iproc == 0:
        subset = grid.decompose(nproc)
    else: 
        subset = None
    subset = comm.scatter(subset, root=0)

    # gather results on rank 0 
    results = _evaluate_misfit([data, greens, misfit, subset])
    results = MPI.COMM_WORLD.gather(results, root=0)


def _evaluate_misfit(args):
    data, greens, misfit, grid = args

    # array to hold misfit values
    results = np.zeros(grid.stop-grid.start)
    count = 0

    for mt in grid:
        #print grid.index

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




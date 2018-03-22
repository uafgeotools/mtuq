
import numpy as np
from util.util import timer, timer_mpi



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


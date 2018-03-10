
import numpy as np

from util.util import timer, timer_mpi



@timer
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


@timer_mpi
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

    # evaluate misfit
    results = _evaluate_misfit([data, greens, misfit, subset])

    # gather results from all processes
    results = comm.gather(results, root=0)




def _evaluate_misfit(args):
    data, greens, misfit, grid = args

    # array to hold misfit values
    results = np.zeros(grid.size)
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




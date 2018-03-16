
import numpy as np

from util.util import timer, timer_mpi



@timer
def grid_search_serial(data, greens, misfit, grid, write_netcdf=True):
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
            func, dat, syn = misfit[key], data[key], synthetics[key]
            results[count] += func(dat, syn)

        count += 1

    if write_netcdf:
        grid.save(_event_name(data), {'misfit': results})

    return grid.get(results.argmin())


@timer_mpi
def grid_search_mpi(data, greens, misfit, grid, write_netcdf=True):
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
    results = _gather(results, comm)

    if iproc==0:
        if write_netcdf:
            grid.save(_event_name(data), {'misfit': results})

        return grid.get(results.argmin())



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
            func, dat, syn = misfit[key], data[key], synthetics[key]
            results[count] += func(dat, syn)

        count += 1

    return results


def _event_name(data):
    data = data[data.keys()[0]]

    if hasattr(data, 'id'):
        return data.id+'.h5'
    else:
        return 'output.h5'


def _gather(results, comm):
    results = comm.gather(results, root=0)
    return np.asarray(results).flatten()



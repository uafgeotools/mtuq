
import numpy as np

from mtuq.mt.maps.tape2015 import tt152cmt
from mtuq.util.util import is_mpi_env, Struct



def grid_search_serial(data, greens, misfit, grid):
    """ Grid search over moment tensor parameters
    """
    categories = data.keys()
    misfit_values = np.zeros(grid.size)

    for _i in range(grid.size):
        print _i

        # gets the i-th moment tensor in grid
        mt = grid.get(_i)

        # generate_synthetics
        synthetics = {}
        for key in categories:
            synthetics[key] = greens[key].get_synthetics(mt)

       # evaluate misfit function
        for key in categories:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            misfit_values[_i] += chi(dat, syn)

    print '\nmin:', min(misfit_values)



def grid_search_mpi(data, greens, misfit, grid):
    from schwimmbad import MPIPool, SerialPool

    # set up parallel queue
    if is_mpi_env():
        pool = MPIPool()
    else:
        pool = SerialPool()

    # constuct arguments list
    tasks = []
    for _i in range(grid.size):
        tasks += [(data, greens, misfit, grid, _i)]

    # evaluate misfit function over all grid points
    misfit_values = pool.map(
        _evalutate_misfit,
        tasks)

    pool.close()


def grid_search_pool(data, greens, misfit, grid, nproc):
    from multiprocessing import Pool
    pool = Pool(nproc)

    # constuct arguments list
    tasks = []
    for indices in _partition(grid, nproc):
        tasks += [(data, greens, misfit, grid, indices)]

    # evaluate misfit function over all grid points
    misfit_values = pool.map(
        _evalutate_misfit,
        tasks)

    pool.close()


def _evalutate_misfit(*args):
    data, greens, misfit, grid, indices = args
    categories = data.keys()

    _misfit_values = []
    for _i in indices:
        print _i
        # gets the i-th moment tensor in grid
        mt = grid.get(_i)

        # generate syntethics
        synthetics = {}
        for key in categories:
            synthetics[key] = greens[key].get_synthetics(mt)

        _misfit_value = 0.
        for key in categories:
            chi, dat, syn = misfit[key], data[key], synthetics[key]
            _misfit_value += chi(dat, syn)
        _misfit_values += [_misfit_value]


def _partition(grid, nproc):
    npts = grid.size/nproc
    indices = []
    for iproc in range(nproc):
        indicies += [np.linspace(0,npts-1)+iproc*npts]
    return indices




import numpy as np
import time
from mtuq.util import iterable, timer


def grid_search(data, greens, misfit, origins, sources, **kwargs):
    """ Evaluates misfit over origin and source grids

    If invoked from within an MPI environment, the grid is divided evenly 
    between MPI processes. Scalability is usually fairly good, as discussed in 
    `docs/`. If not invoked from an MPI environment, this function reduces to 
    the private helper function _grid_search_serial
    """
    if _is_mpi_env():
        # To carry out a grid search over multiple MPI processes, we decompose
        # the grid into subsets and scatter using MPI. Each process then runs
        # _grid_search_serial on it assigned subset
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        iproc, nproc = comm.rank, comm.size

        if iproc == 0:
            sources = sources.decompose(nproc)
        sources = comm.scatter(sources, root=0)

        if iproc == 0:
            if 'verbose' not in kwargs:
                kwargs['verbose'] = True
        else:
            kwargs['verbose'] = False

    return _grid_search_serial(
        data, greens, misfit, origins, sources, **kwargs)


@timer
def _grid_search_serial(data, greens, misfit, origins, sources, verbose=True):
    """ Grid search helper function

    When `grid_search` is called from within an MPI environment, each MPI 
    process executes this helper function
    """
    results = []
    for origin in iterable(origins):
        results += [misfit(data, greens.select(origin), sources)]
    return np.concatenate(results)


def _is_mpi_env():
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.Get_size()>1:
        return True
    else:
        return False




import numpy as np
from mtuq.util import iterable, timer



def grid_search(data, greens, misfit, origins, sources, 
    allgather=True, verbose=True):

    """ Evaluates misfit over origin and source grids

    If invoked from an MPI environment, the grid is partitioned between
    processes and each process runs grid_search_serial. Otherwise, reduces to
    ``grid_search_serial``
    """
    if _is_mpi_env():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        iproc, nproc = comm.rank, comm.size

        # To carry out a grid search over multiple processes, partition the
        # grid into subsets and scatter. Each process then runs
        # grid_search_serial on it assigned subset
        if iproc == 0:
            sources = sources.partition(nproc)
        sources = comm.scatter(sources, root=0)

        # Now adjust keyword arguments
        if iproc != 0:
            verbose = False

    results = grid_search_serial(
        data, greens, misfit, origins, sources, verbose)

    if allgather and _is_mpi_env():
        # Distribute the results to all processes
        return np.concatenate(comm.allgather(results))
    else:
        return results



@timer
def grid_search_serial(data, greens, misfit, origins, sources, 
    verbose=True):
    """ Evaluates misfit over origin and source grids 
    (serial implementation)
    """
    results = []
    for origin in iterable(origins):
        results += [misfit(data, greens.select(origin), sources)]
    return np.concatenate(results, axis=1)



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



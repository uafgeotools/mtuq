
import numpy as np
from mtuq.grid_search import serial


def grid_search(data, greens, misfit, origins, sources, 
    allgather=True, **kwargs):

    """ Evaluates misfit over origin and source grids (MPI implementation)

    If invoked from an MPI environment, the grid is partitioned between
    processes and each process runs serial.grid_search

    If not invoked from an MPI environment, this function reduces to 
    serial.grid_search
    """
    if _is_mpi_env():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        iproc, nproc = comm.rank, comm.size

        # To carry out a grid search over multiple processes, partition the
        # grid into subsets and scatter. Each process then runs
        # serial.grid_search on it assigned subset
        if iproc == 0:
            sources = sources.decompose(nproc)
        sources = comm.scatter(sources, root=0)

        # Now adjust keyword arguments
        if iproc == 0:
            kwargs.setdefault('verbose', True)
        else:
            kwargs['verbose'] = False

    results = serial.grid_search(
        data, greens, misfit, origins, sources, **kwargs)

    if allgather and _is_mpi_env():
        # Distribute the results to all processes
        return np.concatenate(comm.allgather(results))
    else:
        return results


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




import time
from mtuq.grid_search import serial



def grid_search(*args, **kwargs):
    return _mpi_wrapper(
        serial.grid_search, *args, **kwargs)



def _mpi_wrapper(grid_search, data, greens, misfit, sources, *args, **kwargs):
    """ Parallelizes serial grid search function

    To carry out a grid search over multiple MPI processes, we decompose the
    grid into subsets and scatter using MPI. Each process then runs
    grid_search_serial on it assigned subset
    """
    if not _is_mpi_env():
        raise EnvironmentError

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    if not hasattr(sources, 'decompose'):
        raise TypeError

    if iproc == 0:
        subset = grid.decompose(nproc)
    else:
        subset = None
    subset = comm.scatter(subset, root=0)

    if iproc == 0:
        if 'verbose' not in kwargs:
            kwargs['verbose'] = True
    else:
        kwargs['verbose'] = False

    return grid_search(
        data, greens, misfit, subset, *args, **kwargs)


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





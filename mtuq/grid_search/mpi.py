
import time
from mtuq.grid_search import serial


def is_mpi_env():
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


def timer_mpi(func):
    """ Decorator for measuring execution time in mpi environment
    """
    def timed_func(*args, **kwargs):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        if comm.rank==0:
            start_time = time.time()

        output = func(*args, **kwargs)
        comm.barrier()

        if comm.rank==0:
            _elapsed_time = time.time() - start_time
            print '  Elapsed time (s): %f\n' % _elapsed_time

        return output

    return timed_func


@timer_mpi
def grid_search_mt(data, greens, misfit, grid):
    """  Parallel grid search

    To carry out a grid search over multiple MPI processes, we decompose the
    grid into subsets and scatter using MPI. Each process then runs
    grid_search_serial on it assigned subset
    """
    if not is_mpi_env():
        raise EnvironmentError

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    iproc, nproc = comm.rank, comm.size

    if iproc == 0:
        subset = grid.decompose(nproc)
    else:
        subset = None
    subset = comm.scatter(subset, root=0)

    return serial.grid_search_mt(data, greens, misfit, subset, verbose=False)


def grid_search_mt_origin(data, greens, misfit, grid):
    raise NotImplementedError



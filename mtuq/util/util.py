
from os.path import abspath, join
import time


class AttribDict(dict):
    """ Dictionary with both keyword and attribute access
    """
    def __init__(self, *args, **kwargs):
        super(AttribDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Struct(dict):
    """ Dictionary with both keyword and attribute access
    """
    def __init__(self, *args, **kwargs):
        super(Struct, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


def iterable(arg):
    if not isinstance(arg, (list, tuple)):
        return [arg]
    else:
        return arg


def timer(func):
    """ Decorator for measuring execution time
    """
    def timed_func(*args, **kwargs):
        start_time = time.time()

        func(*args, **kwargs)

        _elapsed_time = time.time() - start_time
        print 'Elapsed time:', _elapsed_time

    return timed_func


def timer_mpi(func):
    """ Decorator for measuring execution time in mpi environment
    """
    def timed_func(*args, **kwargs):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        if comm.rank==0:
            start_time = time.time()

        func(*args, **kwargs)

        if comm.rank==0:
            _elapsed_time = time.time() - start_time
            print 'Elapsed time:', _elapsed_time

    return timed_func


def root():
    import mtuq
    return abspath(join(mtuq.__path__[0], '..'))


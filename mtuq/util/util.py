
from os.path import abspath, join
import csv
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


def parse_cap_weight_file(filename):
    """ Parses CAP-style weight file
    """
    weights = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            id = '.'.join(row[0].split('.')[1:4])
            weights[id] = [float(w) for w in row[1:]]

    return weights


def timer(func):
    """ Decorator for measuring execution time
    """
    if is_mpi_env():
        return func

    def timed_func(*args, **kwargs):
        start_time = time.time()

        output = func(*args, **kwargs)

        _elapsed_time = time.time() - start_time
        print '  Elapsed time: %f\n' % _elapsed_time

        return output

    return timed_func


def timer_mpi(func):
    """ Decorator for measuring execution time in mpi environment
    """
    def timed_func(*args, **kwargs):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        if comm.rank==0:
            start_time = time.time()

        output = func(*args, **kwargs)

        if comm.rank==0:
            _elapsed_time = time.time() - start_time
            print '  Elapsed time: %f\n' % _elapsed_time

        return output

    return timed_func


def root():
    import mtuq
    return abspath(join(mtuq.__path__[0], '..'))


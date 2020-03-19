
from os.path import abspath, join
from retry import retry

import copy
import csv
import time
import numpy as np
import obspy
import re
import uuid
import warnings
import zipfile


try:
    from urllib import URLopener
except ImportError:
    from urllib.request import URLopener


class AttribDict(obspy.core.util.attribdict.AttribDict):
    pass


def asarray(x):
    """ Numpy array typecast
    """
    return np.array(x, dtype=np.float64, ndmin=1, copy=False)


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
    """ Simple list typecast
    """
    from mtuq.grid import Grid, UnstructuredGrid
    if not isinstance(arg, (list, tuple, Grid, UnstructuredGrid)):
        return [arg]
    else:
        return arg


def remove_list(list1, list2):
    """ Removes all items of list2 from list1
    """
    for item in list2:
        try:
            list1.remove(item)
        except ValueError:
            pass
    return list1


def replace(string, *args):
    narg = len(args)

    iarg = 0
    while iarg < narg:
        string = re.sub(args[iarg], args[iarg+1], string)
        iarg += 2
    return string


def timer(func):
    """ Decorator for measuring execution time; prints elapsed time to
        standard output
    """
    def timed_func(*args, **kwargs):
        start_time = time.time()

        output = func(*args, **kwargs)

        if kwargs.get('verbose', True):
            _elapsed_time = time.time() - start_time
            print('  Elapsed time (s): %f\n' % _elapsed_time)

        return output

    return timed_func


def basepath():
    """ MTUQ base directory
    """
    import mtuq
    return abspath(join(mtuq.__path__[0], '..'))


def fullpath(*args):
    """ Prepends MTUQ base diretory to given path
    """
    return join(basepath(), *args)


def timer(func):
    """ Decorator for measuring execution time
    """
    def timed_func(*args, **kwargs):
        if kwargs.get('verbose', True):
            start_time = time.time()
            output = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print('  Elapsed time (s): %f\n' % elapsed_time)
            return output
        else:
            return func(*args, **kwargs)

    return timed_func


def unzip(filename):
    parts = filename.split('.')
    if parts[-1]=='zip':
        dirname = '.'.join(parts[:-1])
    else:
        dirname = filename
        filename += '.zip'

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(dirname)
    zip_ref.close()

    return dirname


def warn(*args, **kwargs):
    try:
        from mpi4py import MPI
        comm = MPI_WORLD.COMM
        if comm.rank==0:
           warnings.warn(*args, **kwargs)
    except:
       warnings.warn(*args, **kwargs)


@retry(Exception, tries=4, delay=2, backoff=2)
def urlopen_with_retry(url, filename):
    opener = URLopener()
    opener.retrieve(url, filename)


def url2uuid(url):
    """ Converts a url to a uuid string
    """
    namespace = uuid.NAMESPACE_URL
    name = url
    return uuid.uuid5(namespace, name)


class ProgressBar(object):
    def __init__(self, start, stop, interval=25):
        assert (0 <= start)
        assert (start <= stop)
        assert (1 <= interval <= 50.)

        self.start_index = start
        self.stop_index = stop
        self.next_index = start
        self.percent_interval = round(interval)
        self.index_interval = round(interval)/100.*stop

        self._index = start


    def __call__(self):
        if self._index == self.next_index:
            print("  about %d percent finished" % (100*self._index/self.stop_index))
            self.next_index += self.index_interval

        self._index += 1




#
# make central objects easily accessible
#

from mtuq.dataset import Dataset
from mtuq.greens_tensor import GreensTensorList
from mtuq.event import Origin, Force, MomentTensor
from mtuq.station import Station


#
# register I/O functions
#

from pkg_resources import iter_entry_points
from mtuq.io.clients.syngine import get_greens_tensors


def _greens_tensor_clients():
    clients = {}
    for entry_point in iter_entry_points('greens_tensor_clients'):
        clients[entry_point.name] = entry_point.load()
    return clients


def open_db(path_or_url='', format='', **kwargs):
    """ Opens databse containing Green's functions

    Once opened, ``GreensTensor`` objects can be generated using the
    ``get_greens_tensor`` method of the database.
    """
    format = format.upper()
    return _greens_tensor_clients()[format](path_or_url=path_or_url, **kwargs)


def _readers():
    readers = {}
    for entry_point in iter_entry_points('readers'):
        readers[entry_point.name] = entry_point.load()
    return readers


def read(path='', format='', **kwargs):
    """ Reads waveform files into an MTUQ Databse

    The read() function parses multiple waveforms corresponding to a single 
    seismic event. 

    It returns an Dataset object, a list-like container in which each item
    corresponds to a station and all items correspond to a single event.
    """
    format = format.upper()
    return _readers()[format](path, **kwargs)


#
# ensures MPI calls will continue to work, even if users create bound class
# methods
#

import copy_reg
import types

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)





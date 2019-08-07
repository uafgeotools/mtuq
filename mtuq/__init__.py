
#
# make central objects easily accessible
#

from mtuq.dataset import Dataset
from mtuq.greens_tensor.base import GreensTensor, GreensTensorList
from mtuq.event import Origin, Force, MomentTensor
from mtuq.station import Station
from mtuq.wavelet import Wavelet


#
# register I/O functions
#

from pkg_resources import iter_entry_points
from mtuq.io.clients.syngine import download_greens_tensors


def _greens_tensor_clients():
    clients = {}
    for entry_point in iter_entry_points('greens_tensor_clients'):
        clients[entry_point.name] = entry_point.load()
    return clients


def open_db(path_or_url='', format='', **kwargs):
    """ Opens database containing Green's functions

    Returns an client that can then be used to generate ``GreensTensor``,
    as in the following example

    .. rubric:: Example

    .. code::

        db = open_db('axisem_database.nc', format='AxiSEM')

        greens = db.get_greens_tensors(stations, origin)

        
    """
    format = format.upper()
    return _greens_tensor_clients()[format](path_or_url=path_or_url, **kwargs)


def _readers():
    readers = {}
    for entry_point in iter_entry_points('readers'):
        readers[entry_point.name] = entry_point.load()
    return readers


def read(path='', format='', **kwargs):
    """ Reads waveform files from disk

    Reads files from specified path or url, parses waveform data based on the
    specified file format, and returns an ``mtuq.Dataset``, a container in
    which each element is an ObsPy stream corresponding to a single station
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





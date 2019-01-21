
#
# make central objects easily accessible
#

from mtuq.dataset import Dataset
from mtuq.greens_tensor import GreensTensor, GreensTensorList
from mtuq.event import Origin, MomentTensor, Force
from mtuq.station import Station


#
# register I/O functions
#

from pkg_resources import iter_entry_points


def _greens_tensor_clients():
    clients = {}
    for entry_point in iter_entry_points('greens_tensor_clients'):
        clients[entry_point.name] = entry_point.load()
    return clients

def open_db(format='', **kwargs):
    format = format.upper()
    return _greens_tensor_clients()[format](**kwargs)

def _readers():
    readers = {}
    for entry_point in iter_entry_points('readers'):
        readers[entry_point.name] = entry_point.load()
    return readers

def read(path, format='', **kwargs):
    format = format.upper()
    return _readers()[format](path, **kwargs)


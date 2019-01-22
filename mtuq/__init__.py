
#
# make central objects easily accessible
#

from mtuq.dataset import Dataset
from mtuq.greens_tensor import GreensTensor, GreensTensorList
from mtuq.event import Origin, Force, MomentTensor
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


def open_db(path='', format='', **kwargs):
    """ Opens databse containing Green's functions

    Once opened, ``GreensTensor`` objects can be generated using the
    ``get_greens_tensor`` method of the database.
    """
    format = format.upper()
    return _greens_tensor_clients()[format](path, **kwargs)


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




from mtuq.greens_tensor import GreensTensorList
from mtuq.util import iterable



class Client(object):
    """ Abstract base class for database or web service clients

    Details regarding how the GreenTensors are created--whether they are 
    downloaded, read from disk, or computed on-the-fly--are deferred to the 
    subclass.
    """

    def __init__(self, path_or_url='', **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


    def get_greens_tensors(self, stations=[], origins=[]):
        """ Reads Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to the
        a (station, origin) pair from the given lists

        :param stations: List of ``mtuq.Station`` objects
        :param origins: List of ``mtuq.Origin`` objects
        """
        tensors = []
        for origin in iterable(origins):
            for station in iterable(stations):
                tensors += [self._get_greens_tensor(station, origin)]

        return GreensTensorList(tensors)


    def _get_greens_tensor(self, station=None, origin=None):
        raise NotImplementedError("Must be implemented by subclass")



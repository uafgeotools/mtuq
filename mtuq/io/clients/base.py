
from mtuq.greens_tensor import maGreensTensorList
from mtuq.util import iterable



class Client(object):
    """ Abstract base class for database or web service clients

    .. code:

        db = mtuq.greens.open_db(path, format=format, **kwargs)

        greens_tensors = db.read(stations, origin)

    In the first step, the user supplies input arguments, which vary
    depending on the subclass

    In the second step, the user supplies a list of stations and the origin
    locations and times. GreensTensors are then created for all the
    corresponding station-origin pairs. Details regarding how the GreenTensors 
    are created--whether they are downloaded, read from disk, or computed 
    on-the-fly--are deferred to the subclass.
    """

    def __init__(self, path_or_url='', **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


    def get_greens_tensors(self, stations=[], origins=[]):
        """ Reads Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to the
        a station-origin pair from the given list

        :param stations: List of station objects
        :param origin: List of origin objects
        :rtype: mtuq.greens_tensor.GreensTensorList
        """
        iterator = zip(iterable(stations), iterable(origins))

        greens_tensors = []
        for station, origin in iterator:
            greens_tensors += [self._get_greens_tensor(station, origin)]

        return maGreensTensorList(greens_tensors)


    def _get_greens_tensor(self, station=None, origin=None):
        raise NotImplementedError("Must be implemented by subclass")




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


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Reads Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to the
        a (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)

        """
        origins = iterable(origins)
        stations = iterable(stations)
        ni = len(origins)
        nj = len(stations)

        tensors = []
        for _i, origin in enumerate(origins):
            if verbose:
                if len(origins) > 1:
                    print("  reading %d of %d" % (_i+1, ni))
                    print("  origin latitude: %.1f" % origin.latitude)
                    print("  origin longitude: %.1f" % origin.longitude)
                    print("  origin depth (km): %d" % int(origin.depth_in_m/1000.))
                    print("")

            for _j, station in enumerate(stations):
                tensors += [self._get_greens_tensor(station, origin)]

        return GreensTensorList(tensors)


    def _get_greens_tensor(self, station=None, origin=None):
        raise NotImplementedError("Must be implemented by subclass")



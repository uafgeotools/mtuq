
import mtuq.greens_tensor
from mtuq.util.util import iterable
from obspy.geodetics import gps2dist_azimuth



class GreensTensor(mtuq.greens_tensor.GreensTensor):
    pass


class Client(object):
    """ Abstract base class for database or web service clients

    Details regarding how the GreenTensors are actually created--whether
    they are downloaded on-the-fly or read from a pre-computed database--
    are deferred to the subclass.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


    def get_greens_tensors(self, stations=None, origin=None):
        """ Reads Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to the
        given station and all elements correspond to the given origin

        :type stations: list
        :param stations: List of station metadata dictionaries
        :type origin: obspy.core.event.Origin
        :param origin: Event metadata dictionary
        :rtype: mtuq.greens_tensor.GreensTensorList
        """
        greens_tensors = mtuq.greens_tensor.GreensTensorList()

        for station in iterable(stations):
            # if hypocenter is an inversion parameter, then the values 
            # calculated below will in general differ from preliminary_distance
            # and preliminary_azimuth
            station.distance_in_m, station.azimuth, _ = gps2dist_azimuth(
                origin.latitude,
                origin.longitude,
                station.latitude,
                station.longitude)

            greens_tensors += self._get_greens_tensor(
                station=station, origin=origin)

        return greens_tensors


    def _get_greens_tensor(self, station=None, origin=None):
        raise NotImplementedError("Must be implemented by subclass")



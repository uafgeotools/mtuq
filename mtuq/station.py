
from obspy.core import Stats, UTCDateTime
from obspy.geodetics import gps2dist_azimuth


class Station(Stats):
    """ Station metadata object

    Holds the following station metadata
    - network, station, and location codes
    - preliminary event location and origin time estimates
    - time discretization information for traces recorded at the station

    At the beginning of an inversion, MTUQ requires preliminary estimates for
    event location and depth. Attributes with the suffix "preliminary"
    represent these initial estimates, which are usually based on catalog 
    solutions stored in IRIS waveform metadata.

    Time discretization attributes ``npts``, ``delta``, ``starttime``, and
    ``endtime`` are inherited from the ObsPy base class. This works because we
    always check in ``mtuq.io.readers`` that all traces at a given station have 
    the same time discretization.

    .. note::

        We use ``obspy.core.traces.Stats`` as a base class because
        ``obspy.core.inventory.station.Station`` is more complex than we need.
        (All we require is an attribute-dictionary.)
    """
    readonly = [
        'endtime',
        #'preliminary_distance_in_m',
        #'preliminary_azimuth',
        #'preliminary_backazimuth',
        ]

    defaults = {
        'sampling_rate': 1.0,
        'delta': 1.0,
        'starttime': UTCDateTime(0),
        'endtime': UTCDateTime(0),
        'npts': 0,
        'calib': 1.0,
        'network': '',
        'station': '',
        'location': '',
        'channel': '',
        'latitude': None,
        'longitude': None,
        'station_depth': 0.,
        'station_elevation': 0.,
        'preliminary_origin_time': None,
        'preliminary_event_latitude': None,
        'preliminary_event_longitude': None,
        'preliminary_event_depth_in_m': 0.,
        'preliminary_distance_in_m': 0.,
        'preliminary_azimuth': 0.,
        'preliminary_backazimuth': 0.,
        }


    def __init__(self, *args, **kwargs):
        super(Station, self).__init__(*args, **kwargs)


    def __setitem__(self, key, value):
        # enforce types
        if key in ['preliminary_origin_time']:
            value = UTCDateTime(value)

        super(Station, self).__setitem__(key, value)

        # set readonly values
        if not (self.latitude and self.longitude 
            and self.preliminary_event_latitude
            and self.preliminary_event_longitude):
            return

        if key in ['latitude',
                   'longitude',
                   'preliminary_event_latitude', 
                   'preliminary_event_longitude']:

            distance_in_m, azimuth, backazimuth = gps2dist_azimuth(
                self.preliminary_event_latitude,
                self.preliminary_event_longitude,
                self.latitude,
                self.longitude)

            self.__dict__['preliminary_distance_in_m'] = distance_in_m
            self.__dict__['preliminary_azimuth'] = azimuth
            self.__dict__['preliminary_azimuth'] = backazimuth




import obspy
from obspy.core import UTCDateTime
from obspy.core.util import AttribDict
from obspy.geodetics import gps2dist_azimuth


class Station(AttribDict):
    """ Station metadata object

    Holds the following information
    - latitude, longitude, depth, and elevation
    - network, station, and location codes

    Optionally, also includes 
    - time discretization attributes, which can be useful if all traces recorded
      at the given station have the same time sampling 
    - preliminary origin time and location estimates
    """

    _include_temporal = True
    _include_origin = True


    defaults = {
        'latitude': None,
        'longitude': None,
        'depth_in_m': None,
        'elevation_in_m': None,
        'network': '',
        'station': '',
        'location': '',
        'channel': '',
        }

    readonly = []

    _default_keys = [
        'latitude',
        'longitude',
        'depth_in_m',
        'elevation_in_m',
        'network',
        'station',
        'location',
        'channel',
        ]

    _geographic_keys = [
        'latitude',
        'longitude',
        ]


    # optionally, add time discretization attributes
    if _include_temporal:
        defaults.update({
            'sampling_rate': 1.0,
            'delta': 1.0,
            'starttime': UTCDateTime(0),
            'endtime': UTCDateTime(0),
            'npts': 0,
            })

        readonly.extend([
            'endtime',
            ])

        _temporal_keys = [
            'sampling_rate',
            'delta',
            'starttime',
            'endtime',
            'npts',
            ]


    # optionally, add preliminary origin attributes
    if _include_origin:
        defaults.update({
            'preliminary_origin_time': None,
            'preliminary_event_latitude': None,
            'preliminary_event_longitude': None,
            'preliminary_event_depth_in_m': None,
            'preliminary_distance_in_m': None,
            'preliminary_azimuth': None,
            'preliminary_backazimuth': None,
            })

        readonly.extend([
            'preliminary_distance_in_m',
            'preliminary_azimuth',
            'preliminary_backazimuth',
            ])

        _geographic_keys += [
            'preliminary_event_latitude',
            'preliminary_event_longitude',
            ]

        _origin_keys = [
            'preliminary_origin_time',
            'preliminary_event_latitude',
            'preliminary_event_longitude',
            'preliminary_event_depth_in_m',
            'preliminary_distance_in_m',
            'preliminary_azimuth',
            'preliminary_backazimuth',
            ]


    def __init__(self, *args, **kwargs):
        super(Station, self).__init__(*args, **kwargs)


    def __setitem__(self, key, value):
        if self._include_temporal and key in self._temporal_keys:
            self._set_temporal_item(key, value)

        elif self._include_origin and key in self._origin_keys:
            self._set_origin_item(key, value)

        elif isinstance(value, dict):
            super(Station, self).__setitem__(key, AttribDict(value))

        else:
            super(Station, self).__setitem__(key, value)

        if self._include_origin and key in self._geographic_keys:
            self._refresh()


    def _set_temporal_item(self, key, value):
        # adapted from obspy.core.trace.Stats

        if key == 'npts':
            value = int(value)

        elif key == 'sampling_rate':
            value = float(value)

        elif key == 'starttime':
            value = UTCDateTime(value)

        elif key == 'delta':
            key = 'sampling_rate'
            try:
                value = 1.0 / float(value)
            except ZeroDivisionError:
                value = 0.

        super(Station, self).__setitem__(key, value)

        try:
            self.__dict__['delta'] = 1.0 / float(self.sampling_rate)
        except ZeroDivisionError:
            self.__dict__['delta'] = 0.

        if self.npts > 0:
            self.__dict__['endtime'] = self.starttime +\
                float(self.npts-1)*self.delta
        else:
            self.__dict__['endtime'] = self.starttime


    def _set_origin_item(self, key, value):
        if value is None:
            pass

        elif key in ['preliminary_origin_time']:
            value = UTCDateTime(value)

        elif key in [
            'preliminary_event_latitude',
            'preliminary_event_longitude',
            'preliminary_event_depth_in_m',
            ]:
            value = float(value)

        super(Station, self).__setitem__(key, value)


    def _refresh(self):
        for key in self._geographic_keys:
            if self.__dict__[key] is None:
                return

        distance_in_m, azimuth, backazimuth = gps2dist_azimuth(
            self.preliminary_event_latitude,
            self.preliminary_event_longitude,
            self.latitude,
            self.longitude)

        self.__dict__['preliminary_distance_in_m'] = distance_in_m
        self.__dict__['preliminary_azimuth'] = azimuth
        self.__dict__['preliminary_backazimuth'] = backazimuth





import obspy
from obspy.core import UTCDateTime
from obspy.core.util import AttribDict
from obspy.geodetics import gps2dist_azimuth


class Station(AttribDict):
    """Station metadata object

    Holds the following information

    - latitude and longitude
    - depth and elevation
    - network, station, and location codes

    .. note::

        Each supported file format has a corresponding reader that creates
        Station objects from file metadata (see ``mtuq.io.readers``).

    """


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
    _refresh_keys = []


    if True:
        # optional time discretization attributes
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

        _refresh_keys.extend([
            'sampling_rate',
            'delta',
            'starttime',
            'endtime',
            'npts',
            ])


    def __init__(self, *args, **kwargs):
        super(Station, self).__init__(*args, **kwargs)


    def __setitem__(self, key, value):
        if key in self._refresh_keys:
            self._refresh(key, value)

        elif isinstance(value, dict):
            super(Station, self).__setitem__(key, AttribDict(value))

        else:
            super(Station, self).__setitem__(key, value)


    def _refresh(self, key, value):
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




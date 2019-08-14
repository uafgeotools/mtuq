
try:
    import instaseis
except:
    pass
import obspy
import numpy as np

from os.path import basename
from mtuq.greens_tensor.axisem_netcdf import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import get_distance_in_deg, resample



class Client(ClientBase):
    """ 
    AxiSEM NetCDF database client (based on instaseis)


    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.axisem_netcdf import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)

    """

    def __init__(self, path_or_url='', model='', kernelwidth=12):
        if not path_or_url:
            raise Exception
        self.db = instaseis.open_db(path_or_url)
        self.kernelwidth=12

        if not model:
            model = path_or_url
        self.model = model


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Reads Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        :param stations: List of ``mtuq.Station`` objects
        :param origins: List of ``mtuq.Origin`` objects
        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        distance_in_deg = get_distance_in_deg(station, origin)

        stream = self.db.get_greens_function(
            epicentral_distance_in_degree=distance_in_deg,
            source_depth_in_m=origin.depth_in_m, 
            origin_time=origin.time,
            kind='displacement',
            kernelwidth=self.kernelwidth,
            definition=u'seiscomp')

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # what are the start and end times of the Green's function?
        trace = stream[0]
        t1_old = float(trace.stats.starttime)
        t2_old = float(trace.stats.endtime)
        dt_old = float(trace.stats.delta)

        for trace in stream:
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'syngine',
             ]

        return GreensTensor(traces=[trace for trace in stream],
            station=station, origin=origin, tags=tags)



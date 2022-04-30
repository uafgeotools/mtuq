
try:
    import instaseis
except:
    pass

import obspy
import numpy as np

from obspy.core import Stream
from os.path import basename
from mtuq.greens_tensor.AxiSEM import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import get_distance_in_deg, resample



class Client(ClientBase):
    """ 
    AxiSEM NetCDF database client (based on `instaseis <https://instaseis.net/>`_)


    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.AxiSEM_NetCDF import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

        For instructions on creating AxiSEM NetCDF databases, see
        `AxiSEM user manual - Output wavefields in netcdf format needed for Instaseis
        <https://raw.githubusercontent.com/geodynamics/axisem/master/MANUAL/manual_axisem1.3.pdf>`_

    """

    def __init__(self, path_or_url='', model='', kernelwidth=12,
        include_mt=True, include_force=False):

        if not path_or_url:
            raise Exception
        self.db = instaseis.open_db(path_or_url)
        self.kernelwidth=12

        if not model:
            model = path_or_url
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Reads Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)
        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):

        stream = Stream()

        if self.include_mt:
            stream += self.db.get_greens_function(
                epicentral_distance_in_degree=get_distance_in_deg(station, origin),
                source_depth_in_m=origin.depth_in_m, 
                origin_time=origin.time,
                kind='displacement',
                kernelwidth=self.kernelwidth,
                definition='seiscomp')

        if self.include_force:
            receiver = _get_instaseis_receiver(station)

            for _i, force in enumerate([{'f_r': 1.}, {'f_t': 1.}, {'f_p': 1.}]):
                stream += self.db.get_seismograms(
                    source=_get_instaseis_source(origin, **force),
                    receiver=receiver,
                    components=['Z','R','T'],
                    kind='displacement',
                    kernelwidth=self.kernelwidth)

                stream[-3].stats.channel = "Z"+str(_i)
                stream[-2].stats.channel = "R"+str(_i)
                stream[-1].stats.channel = "T"+str(_i)
            

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
            trace.stats._component = trace.stats.channel[0]

            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'AxiSEM',
             ]

        return GreensTensor(traces=[trace for trace in stream],
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)



#
# utility functions
#

def _get_instaseis_source(origin, **kwargs):
    return instaseis.ForceSource(
        origin.latitude,
        origin.longitude,
        depth_in_m=origin.depth_in_m,
        origin_time=origin.time,
        **kwargs)

def _get_instaseis_receiver(station):
    return instaseis.Receiver(
        station.latitude,
        station.longitude,
        network=station.network,
        station=station.station,
        location=station.location)


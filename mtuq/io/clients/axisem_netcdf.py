
try:
    import instaseis
except:
    pass
import obspy
import numpy as np

from os.path import basename
from mtuq.greens_tensor.axisem_netcdf import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.moment_tensor.basis import change_basis
from mtuq.util.signal import resample
from mtuq.util import m_to_deg



class Client(ClientBase):
    """ 
    Interface to AxiSEM/Instaseis database

    Generates GreenTensorLists via a two-step procedure

    .. code:

        db = mtuq.greens.open_db(path, format='instaseis')

        greens_tensors = db.read(stations, origin)

    In the first step, the user supplies the path or URL to an AxiSEM NetCDF
    output file

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.

    """

    def __init__(self, path_or_url='', kernelwidth=12):
        if not path:
            raise Exception
        try:
            db = instaseis.open_db(path)
        except:
            Exception
        self.db = db
        self.kernelwidth=12


    def _get_greens_tensor(self, station=None, origin=None):
        stream = self.db.get_greens_function(
            epicentral_distance_in_degree=m_to_deg(station.distance_in_m),
            source_depth_in_m=station.depth_in_m, 
            origin_time=origin.time,
            kind='displacement',
            kernelwidth=self.kernelwidth,
            definition=u'seiscomp')
        stream.id = station.id

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # what are the start and end times of the Green's function?
        t1_old = float(origin.time)+float(trace.stats.starttime)
        t2_old = float(origin.time)+float(trace.stats.endtime)
        dt_old = float(trace.stats.delta)

        for trace in stream:
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin)



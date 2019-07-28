
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
from obspy.geodetics import gps2dist_azimuth



class Client(ClientBase):
    """ 
    Interface to AxiSEM/Instaseis database

    .. code:

        db = mtuq.greens.open_db(path, format='instaseis')

        greens_tensors = db.read(stations, origin)

    In the first step, the user supplies the path or URL to an AxiSEM NetCDF
    output file

    In the second step, the user supplies a list of stations and the origin
    locations and times. GreensTensors are then created for all the
    corresponding station-origin pairs.

    """

    def __init__(self, path_or_url='', model='', kernelwidth=12):
        if not path_or_url:
            raise Exception
        self.db = instaseis.open_db(path_or_url)
        self.kernelwidth=12

        if not model:
            model = path_or_url
        self.model = model


    def _get_greens_tensor(self, station=None, origin=None):
        """ 
        Reads a Greens tensor from AxiSEM NetCDF database
        """

        distance_in_m, _, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        stream = self.db.get_greens_function(
            epicentral_distance_in_degree=m_to_deg(distance_in_m),
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

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin, model=self.model)



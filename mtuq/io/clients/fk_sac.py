
import obspy
import numpy as np

from os.path import basename, exists
from mtuq.greens_tensor.fk_sac import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth



class Client(ClientBase):
    """ 
    Interface to FK database of Green's functions

    .. code:

        db = mtuq.greens.open_db(path, model=model, format='FK')

        greens_tensors = db.read(stations, origins)

    In the first step, the user supplies the path to an FK directory tree

    In the second step, the user supplies a list of stations and the origin
    locations and times. GreensTensors are then created for all the
    corresponding station-origin pairs. 

    .. note:

      GreensTensor objects are created by reading precomputed Green's tensors 
      from an FK directory tree.  Such trees contain SAC files organized by model,
      even depth, and event distance and are associated with the FK software 
      package by Lupei Zhu.

    """
    def __init__(self, path_or_url=None, model=None):
        if not path_or_url:
            raise Exception

        if not exists(path_or_url):
            raise Exception

        if not model:
            model = basename(path_or_url)

        # path to fk directory tree
        self.path = path_or_url

        # model from which fk Green's functions were computed
        self.model = model


    def _get_greens_tensor(self, station=None, origin=None):
        """ 
        Reads a Greens tensor from a directory tree organized by model, event
        depth, and event distance
        """
        traces = []

        distance_in_m, _, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        #dep = str(int(round(origin.depth_in_m/1000.)))
        dep = str(int(np.ceil(origin.depth_in_m/1000.)))
        #dst = str(int(round(distance_in_m/1000.)))
        dst = str(int(np.ceil(distance_in_m/1000.)))

        # An FK simulation outputs 12 SAC files each with filename extensions
        # 0,1,2,3,4,5,6,7,8,9,a,b.  The SAC files ending in .2 and .9 contain 
        # only zero data, so we exclude them from the following list. 
        # The order of traces in the list is the order in which CAP stores
        # the time series.
        extensions = [
            '8','5',           # t
            'b','7','4','1',   # r
            'a','6','3','0',   # z
            ]

        channels = [
            'TSS', 'TDS',
            'REP', 'RSS', 'RDS', 'RDD',
            'ZEP', 'ZSS', 'ZDS', 'ZDD',
            ]

        for _i, ext in enumerate(extensions):
            trace = obspy.read('%s/%s_%s/%s.grn.%s' %
                (self.path, self.model, dep, dst, ext),
                format='sac')[0]

            trace.stats.channel = channels[_i]

            # what are the start and end times of the Green's function?
            t1_old = float(origin.time)+float(trace.stats.starttime)
            t2_old = float(origin.time)+float(trace.stats.endtime)
            dt_old = float(trace.stats.delta)
            data_old = trace.data

            # resample Green's function
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            # convert from 10^-20 dyne to N^-1
            trace.data *= 1.e-15
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

            traces += [trace]

        return GreensTensor(traces=traces, station=station, origin=origin,
             model=self.model)




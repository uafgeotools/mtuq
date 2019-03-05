
import obspy
import numpy as np

from math import ceil
from os.path import basename, exists
from obspy.core import Stream
from mtuq import GreensTensor as GreensTensorBase
from mtuq.io.greens_tensor.base import Client as ClientBase
from mtuq.util.signal import resample
from mtuq.util.moment_tensor.basis import change_basis



class GreensTensor(GreensTensorBase):
    """
    FK Green's tensor object

    Overloads base class with the mathematical machinery for working with
    FK-style Green's functions

    FK Green's functions describe the impulse response of a horizontally-
    layered medium.  Time series represent vertical, radial, and transverse
    velocity in units of 10^-20*cm*(dyne-cm)^-1 s^-1

    For the vertical and raidal components, there are four associated time 
    series. For the tranverse component, there are two associated time
    series. Thus there are ten independent Green's tensor elements altogether, 
    which is fewer than in the case of a general inhomogeneous medium
    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        self.tags = []
        self.tags += ['type:greens']
        self.tags += ['type:velocity']
        self.tags += ['units:m']


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        return super(GreensTensor, self).get_synthetics(
            change_basis(mt, 1, 2))


    def get_time_shift(self, data, mt, group, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied data
        """
        return super(GreensTensor, self).get_time_shift(
            data,
            change_basis(mt, 1, 2),
            group,
            time_shift_max)


    def _precompute(self):
        """
        Computes rotated time series used in source-weighted linear combinations
 
        The following expressions were obtained using CAP's mt_radiat utility as
        a starting point
        """
        az = np.deg2rad(self.stats.azimuth)

        # array dimensions
        nt = self[0].stats.npts
        nc = len(self.components)
        nr = 9

        G = np.zeros((nc, nr, nt))
        self._tensor = G

        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = self.select(channel="ZSS")[0].data
                ZDS = self.select(channel="ZDS")[0].data
                ZDD = self.select(channel="ZDD")[0].data
                ZEP = self.select(channel="ZEP")[0].data
                G[_i, 0, :] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
                G[_i, 1, :] =  ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
                G[_i, 2, :] =  ZDD/3. + ZEP/3.
                G[_i, 3, :] = -ZSS * np.sin(2*az)
                G[_i, 4, :] = -ZDS * np.cos(az)
                G[_i, 5, :] = -ZDS * np.sin(az)

            elif component=='R':
                RSS = self.select(channel="RSS")[0].data
                RDS = self.select(channel="RDS")[0].data
                RDD = self.select(channel="RDD")[0].data
                REP = self.select(channel="REP")[0].data
                G[_i, 0, :] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
                G[_i, 1, :] =  RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
                G[_i, 2, :] =  RDD/3. + REP/3.
                G[_i, 3, :] = -RSS * np.sin(2*az)
                G[_i, 4, :] = -RDS * np.cos(az)
                G[_i, 5, :] = -RDS * np.sin(az)

            elif component=='T':
                TSS = self.select(channel="TSS")[0].data
                TDS = self.select(channel="TDS")[0].data
                G[_i, 0, :] = -TSS/2. * np.sin(2*az)
                G[_i, 1, :] =  TSS/2. * np.sin(2*az)
                G[_i, 2, :] =  0.
                G[_i, 3, :] =  TSS * np.cos(2*az)
                G[_i, 4, :] = -TDS * np.sin(az)
                G[_i, 5, :] =  TDS * np.cos(az)

            else:
                raise ValueError



class Client(ClientBase):
    """ 
    Interface to FK database of Green's functions

    Generates GreenTensorLists via a two-step procedure
        1) db = mtuq.greens.open_db(path=path, model=model, format='FK')
        2) greens_tensors = db.read(stations, origin)

    In the first step, the user supplies the path to an FK directory tree and 
    the name of the  layered Earth model that was used to generate Green's
    tensors contained in the tree.

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.

    GreensTensorLists are created by reading precomputed Green's tensors from an
    fk directory tree.  Such trees contain SAC files organized by model, event
    depth, and event distance and are associated with the software package FK
    by Lupei Zhu.

    """
    def __init__(self, path=None, model=None):
        if not path:
            raise Exception

        if not exists(path):
            raise Exception

        if not model:
            model = basename(path)

        # path to fk directory tree
        self.path = path

        # model from which fk Green's functions were computed
        self.model = model


    def _get_greens_tensor(self, station=None, origin=None):
        """ 
        Reads a Greens tensor from a directory tree organized by model, event
        depth, and event distance
        """
        traces = []

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        #dep = str(int(round(origin.depth_in_m/1000.)))
        dep = str(int(ceil(origin.depth_in_m/1000.)))
        #dst = str(int(round(station.distance_in_m/1000.)))
        dst = str(int(ceil(station.distance_in_m/1000.)))

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

        return GreensTensor(traces, station, origin)



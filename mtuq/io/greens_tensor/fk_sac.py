
import obspy
import numpy as np
import mtuq.io.greens_tensor.axisem_netcdf

from math import ceil
from os.path import basename, exists
from obspy.core import Stream
from mtuq.util.signal import resample
from mtuq.util.moment_tensor.basis import change_basis


# fk Green's functions represent vertical, radial, and transverse
# velocity time series (units: 10^-20 cm (dyne-cm)^-1 s^-1) 


# fk Green's functions describe the impulse response of a horizontally layered 
# medium. For the vertical and radial components, there are four associated 
# time series. For the tranverse component, there are two associated time 
# series. Thus there are ten independent Green's tensor elements altogether, 
# which is fewer than in the case of a general inhomogeneous medium


# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# Database normally does all the work

DEG2RAD = np.pi/180.



class GreensTensor(mtuq.io.greens_tensor.axisem_netcdf.GreensTensor):
    """
    Elastic Green's tensor object
    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)
        self.tags += ['type:velocity']


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        if not hasattr(self, '_weighted_tensor'):
            self._precompute_weights()

        # CAP/FK uses convention #2 (Aki&Richards)
        mt = change_basis(mt, 1, 2)

        for _i, component in enumerate(self.components):
            # which Green's functions correspond to given component?
            if component=='Z':
                _j=0
            elif component=='R':
                _j=1
            elif component=='T':
                _j=2
            G = self._weighted_tensor[_j]

            # we could use np.dot instead, but speedup appears negligible
            s = self._synthetics[_i].data
            s[:] = 0.
            s += mt[0]*G[:,0]
            s += mt[1]*G[:,1]
            s += mt[2]*G[:,2]
            s += mt[3]*G[:,3]
            s += mt[4]*G[:,4]
            s += mt[5]*G[:,5]

        return self._synthetics


    def _precompute_weights(self):
        """
        Applies weights used in linear combination of Green's functions
 
        See cap/fk documentation for indexing scheme details; here we try to
        follow as closely as possible the cap way of doing things
 
        See also Lupei Zhu's mt_radiat utility
        """
        npts = self[0].stats['npts']
        az = np.deg2rad(self.stats.azimuth)
        self._weighted_tensor = []

        if 'Z' in self.components:
            ZSS = self.select(channel="ZSS")[0].data
            ZDS = self.select(channel="ZDS")[0].data
            ZDD = self.select(channel="ZDD")[0].data
            ZEP = self.select(channel="ZEP")[0].data

            GZ = np.ones((npts, 6))

            GZ[:, 0] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
            GZ[:, 1] =  ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
            GZ[:, 2] =  ZDD/3. + ZEP/3.
            GZ[:, 3] = -ZSS * np.sin(2*az)
            GZ[:, 4] = -ZDS * np.cos(az)
            GZ[:, 5] = -ZDS * np.sin(az)

            self._weighted_tensor += [GZ]


        if 'R' in self.components:
            RSS = self.select(channel="RSS")[0].data
            RDS = self.select(channel="RDS")[0].data
            RDD = self.select(channel="RDD")[0].data
            REP = self.select(channel="REP")[0].data

            GR = np.ones((npts, 6))

            GR[:, 0] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
            GR[:, 1] =  RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
            GR[:, 2] =  RDD/3. + REP/3.
            GR[:, 3] = -RSS * np.sin(2*az)
            GR[:, 4] = -RDS * np.cos(az)
            GR[:, 5] = -RDS * np.sin(az)

            self._weighted_tensor += [GR]


        if 'T' in self.components:
            TSS = self.select(channel="TSS")[0].data
            TDS = self.select(channel="TDS")[0].data

            GT = np.ones((npts, 6))

            GT[:, 0] = -TSS/2. * np.sin(2*az)
            GT[:, 1] =  TSS/2. * np.sin(2*az)
            GT[:, 2] =  0.
            GT[:, 3] =  TSS * np.cos(2*az)
            GT[:, 4] = -TDS * np.sin(az)
            GT[:, 5] =  TDS * np.cos(az)

            self._weighted_tensor += [GT]



class Client(mtuq.io.greens_tensor.base.Client):
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

        # The output of an FK simulation consists of 12 SAC files with single
        # character filename extensions.  The SAC files ending in *.2 and *.9 
        # contain only zero data, so we exclude them from the following list.  
        # The order of traces in the following list is the order in which CAP
        # stores the individual time series.
        extensions = [
            '8','5',           # t
            'b','7','4','1',   # r
            'a','6','3','0',   # z
            ]

        # The filename extensions above correspond to the following instaseis
        # channel names.
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




import instaseis
import obspy
import numpy as np

import mtuq.greens_tensor.base

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.util.geodetics import km2deg
from mtuq.util.signal import resample


# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encounteRED, since
# GreensTensorFactory normally does all the work
ErrorMessage=''


class GreensTensor(mtuq.greens_tensor.base.GreensTensor):
    """
    Elastic Green's tensor object
    """
    def __init__(self, stream, station, origin):
        assert isinstance(stream, obspy.Stream), ValueError(ErrorMessage)
        super(GreensTensor, self).__init__(stream, station, origin)


    def _calculate_weights(self):
        """ See also: 
            test_instaseis.test_get_greens_vs_get_seismogram
        """
        traces = self.greens_tensor
        npts = traces[0].meta['npts']
        az = np.deg2rad(self.station.azimuth)

        TSS = traces.select(channel="TSS")[0].data
        ZSS = traces.select(channel="ZSS")[0].data
        RSS = traces.select(channel="RSS")[0].data
        TDS = traces.select(channel="TDS")[0].data
        ZDS = traces.select(channel="ZDS")[0].data
        RDS = traces.select(channel="RDS")[0].data
        ZDD = traces.select(channel="ZDD")[0].data
        RDD = traces.select(channel="RDD")[0].data
        ZEP = traces.select(channel="ZEP")[0].data
        REP = traces.select(channel="REP")[0].data

        GZ = np.ones((npts, 6))
        GR = np.ones((npts, 6))
        GT = np.ones((npts, 6))

        GZ[:, 0] = ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
        GZ[:, 1] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
        GZ[:, 2] = ZDD/3. + ZEP/3.
        GZ[:, 3] = ZSS * np.sin(2*az)
        GZ[:, 4] = ZDS * np.cos(az)
        GZ[:, 5] = ZDS * np.sin(az)

        GR[:, 0] = RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
        GR[:, 1] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
        GR[:, 2] = RDD/3. + REP/3.
        GR[:, 3] = RSS * np.sin(2*az)
        GR[:, 4] = RDS * np.cos(az)
        GR[:, 5] = RDS * np.sin(az)

        GT[:, 0] = TSS/2. * np.sin(2*az)
        GT[:, 1] = -TSS/2. * np.sin(2*az)
        GT[:, 2] = 0.
        GT[:, 3] = -TSS * np.cos(2*az)
        GT[:, 4] = TDS * np.sin(az)
        GT[:, 5] = -TDS * np.cos(az)

        self._GZ = GZ
        self._GR = GR
        self._GT = GT


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        if not hasattr(self, '_rotated_greens_tensor'):
            self._calculate_weights()

        for _i, channel in enumerate(self.station.channels):
            component = channel[-1].upper()
            if component not in ['Z','R','T']:
                raise Exception("Channels are expected to end in one of the "
                   "following characters: ZRT")
            self._synthetics[_i].meta.channel = component

            s = self._synthetics[_i].data
            # overwrite previous synthetics
            s[:] = 0.

            if component=='Z':
                G = self._GZ
            if component=='R':
                G = self._GR
            if component=='T':
                G = self._GT

            s += mt[2]*G[:,0]
            s += mt[3]*G[:,1]
            s += mt[1]*G[:,2]
            s += mt[5]*G[:,3]
            s += mt[3]*G[:,4]
            s += mt[4]*G[:,5]

        return self._synthetics


    def _preallocate_synthetics(self):
        self._synthetics = Stream()
        for channel in self.station.channels:
            self._synthetics +=\
                Trace(np.zeros(self.greens_tensor[0].stats.npts), self.station)
        self._synthetics.id = self.greens_tensor.id




class GreensTensorFactory(mtuq.greens_tensor.base.GreensTensorFactory):
    def __init__(self, path, kernelwidth=12):
        try:
            db = instaseis.open_db(path)
        except:
            Exception
        self.db = db
        self.kernelwidth=12


    def get_greens_tensor(self, station, origin):
        stream = self.db.get_greens_function(
            epicentral_distance_in_degree=km2deg(station.distance),
            source_depth_in_m=station.depth, 
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

        return GreensTensor(stream, station, origin)


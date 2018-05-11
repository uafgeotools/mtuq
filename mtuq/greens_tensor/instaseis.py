
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
# message is displayed.  In practice this is rarely encountered, since
# Generator normally does all the work
ErrorMessage=''


class GreensTensor(mtuq.greens_tensor.base.GreensTensor):
    """
    Elastic Green's tensor object
    """
    def __init__(self, stream, station, origin):
        assert isinstance(stream, obspy.Stream), ValueError(ErrorMessage)
        super(GreensTensor, self).__init__(stream, station, origin)


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        if not hasattr(self, '_rotated_greens_tensor'):
            self._calculate_weights()

        self.__array[:] = 0.
        self.__array += self._rotated_greens_tensor[:,0]*mt[0]
        self.__array += self._rotated_greens_tensor[:,1]*mt[1]
        self.__array += self._rotated_greens_tensor[:,2]*mt[3]
        self.__array += self._rotated_greens_tensor[:,3]*mt[4]
        self.__array += self._rotated_greens_tensor[:,4]*mt[5]
        return self._synthetics


    def _calculate_weights(self):
        tss = self.greens_tensor.traces[0].data
        zss = self.greens_tensor.traces[1].data
        rss = self.greens_tensor.traces[2].data
        tds = self.greens_tensor.traces[3].data
        zds = self.greens_tensor.traces[4].data
        rds = self.greens_tensor.traces[5].data
        zdd = self.greens_tensor.traces[6].data
        rdd = self.greens_tensor.traces[7].data
        zep = self.greens_tensor.traces[8].data
        rep = self.greens_tensor.traces[9].data

        G_z = self.greens_tensor.traces[0].meta['npts']
        G_r = self.greens_tensor.traces[0].meta['npts'] * 2
        G_t = self.greens_tensor.traces[0].meta['npts'] * 3
        G = np.ones((G_t, 5))

        azimuth = np.deg2rad(self.station.azimuth)

        G[0:G_z, 0] = zss * (0.5) * np.cos(2 * azimuth) - zdd * 0.5
        G[0:G_z, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * azimuth)
        # G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
        G[0:G_z, 2] = zss * np.sin(2 * azimuth)
        G[0:G_z, 3] = -zds * np.cos(azimuth)
        G[0:G_z, 4] = -zds * np.sin(azimuth)

        G[G_z:G_r, 0] = rss * (0.5) * np.cos(2 * azimuth) - rdd * 0.5
        G[G_z:G_r, 1] = -0.5 * rdd - rss * (0.5) * np.cos(2 * azimuth)
        # G[G_z:G_r, 1] =  rdd * (1/3) + rep * (1/3)
        G[G_z:G_r, 2] = rss * np.sin(2 * azimuth)
        G[G_z:G_r, 3] = -rds * np.cos(azimuth)
        G[G_z:G_r, 4] = -rds * np.sin(azimuth)

        G[G_r:G_t, 0] = -tss * (0.5) * np.sin(2 * azimuth)
        G[G_r:G_t, 1] = tss * (0.5) * np.sin(2 * azimuth)
        # G[G_r:G_t, 1] =   0
        G[G_r:G_t, 2] = tss * np.cos(2 * azimuth)
        G[G_r:G_t, 3] = tds * np.sin(2 * azimuth)
        G[G_r:G_t, 4] = -tds * np.cos(2 * azimuth)

        self._rotated_greens_tensor = G


    def _preallocate_synthetics(self):
        """ 
        Creates obspy streams for use by get_synthetics
        """
        npts = self.greens_tensor[0].stats.npts
        self.__array = np.zeros(3*npts)
        self._synthetics = Stream()
        for _i in range(3):
            self._synthetics +=\
                Trace(self.__array[_i*npts:(_i+1)*npts], self.station)
        self._synthetics.id = self.greens_tensor.id


class Generator(mtuq.greens_tensor.base.Generator):
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


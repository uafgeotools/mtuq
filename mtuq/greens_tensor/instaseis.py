
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


# instaseis Green's functions represent vertical, radial, and transverse
# displacement time series
COMPONENTS = ['Z','R','T']


# instaseis Green's function describe the impulse response of a horizontally-
# layered medium. For the vertical and raidal components, there are four
# associated time series. For the tranverse component, there are two associated 
# time series. Thus there ten independent Green's tensor elements altogether, 
# which is fewer than in the case of a general inhomogeneous medium


# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# GreensTensorFactory normally does all the work
ErrorMessage =("A list of 10 traces must be provided, each representing an"
    "indepedent Green's tensor element.")


class GreensTensor(mtuq.greens_tensor.base.GreensTensor):
    """
    Elastic Green's tensor object
    """
    def __init__(self, traces, station, origin):
        super(GreensTensor, self).__init__(traces, station, origin)


    def _calculate_weights(self):
        """
        Calculates weights used in linear combination of Green's functions

        For more information, see

        -   van Driel et al. (2015)
            Instaseis: instant global seismograms
            Solid Earth, 6, 701-717

        -   Minson, Sarah E. and Douglas S. Dreger (2008)
            Stable Inversions for Complete Moment Tensors
            Geophysical Journal International 174 (2): 585-592

        -   github.com/krischer/instaseis/instaseis/tests/
            test_instaseis.py::test_get_greens_vs_get_seismogram
        """
        npts = self[0].meta['npts']
        az = np.deg2rad(self.meta.azimuth)

        TSS = self.select(channel="TSS")[0].data
        ZSS = self.select(channel="ZSS")[0].data
        RSS = self.select(channel="RSS")[0].data
        TDS = self.select(channel="TDS")[0].data
        ZDS = self.select(channel="ZDS")[0].data
        RDS = self.select(channel="RDS")[0].data
        ZDD = self.select(channel="ZDD")[0].data
        RDD = self.select(channel="RDD")[0].data
        ZEP = self.select(channel="ZEP")[0].data
        REP = self.select(channel="REP")[0].data

        GZ = np.ones((npts, 6))
        GR = np.ones((npts, 6))
        GT = np.ones((npts, 6))

        GZ[:, 0] =  ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
        GZ[:, 1] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
        GZ[:, 2] =  ZDD/3. + ZEP/3.
        GZ[:, 3] =  ZSS * np.sin(2*az)
        GZ[:, 4] =  ZDS * np.cos(az)
        GZ[:, 5] =  ZDS * np.sin(az)

        GR[:, 0] =  RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
        GR[:, 1] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
        GR[:, 2] =  RDD/3. + REP/3.
        GR[:, 3] =  RSS * np.sin(2*az)
        GR[:, 4] =  RDS * np.cos(az)
        GR[:, 5] =  RDS * np.sin(az)

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

        if not hasattr(self, '_GZ'):
            self._calculate_weights()

        for _i, channel in enumerate(self.meta.channels):
            component = channel[-1].upper()
            if component not in COMPONENTS:
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
                # the negative sign is needed because of inconsistent
                # instaseis and syngine moment tensor conventions?
                G = -self._GT


            # Order of terms expected by syngine URL parser according to 
            # IRIS documentation:
            #    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
            #
            # Relations given in instaseis/tests/test_instaseis.py:
            #    m_tt=Mxx, m_pp=Myy, m_rr=Mzz, m_rt=Mxz, m_rp=Myz, m_tp=Mxy
            #
            # Relations suggested by mtuq/tests/unittest_greens_tensor_syngine.py
            # (note sign differences):
            #    m_tt=Mxx, m_pp=Myy, m_rr=Mzz, m_rt=M-xz, m_rp=Myz, m_tp=-Mxy
            Mxx =  mt[1]
            Myy =  mt[2]
            Mzz =  mt[0]
            Mxy = -mt[5]
            Mxz = -mt[3]
            Myz =  mt[4]

            s += Mxx*G[:,0]
            s += Myy*G[:,1]
            s += Mzz*G[:,2]
            s += Mxy*G[:,3]
            s += Mxz*G[:,4]
            s += Myz*G[:,5]

        return self._synthetics


    def _preallocate_synthetics(self):
        self.meta.npts = self[0].stats.npts
        self._synthetics = Stream()
        for channel in self.meta.channels:
            self._synthetics +=\
                Trace(np.zeros(self[0].stats.npts), self.meta)
        self._synthetics.id = self.id



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

        traces = [trace for trace in stream]
        return GreensTensor(stream, station, origin)


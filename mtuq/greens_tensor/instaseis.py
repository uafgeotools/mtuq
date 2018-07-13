
import instaseis
import obspy
import numpy as np
import mtuq.greens_tensor.base

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists
from obspy.core import Stream, Trace
from scipy.signal import fftconvolve
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

        self._rotated_tensor = []
        self._rotated_tensor += [GZ]
        self._rotated_tensor += [GR]
        # the negative sign is needed because of inconsistent moment tensor
        #  conventions?
        self._rotated_tensor += [-GT]


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
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

        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        if not hasattr(self, '_rotated_tensor'):
            self._calculate_weights()

        for _i, component in enumerate(self.meta.components):
            # which Green's functions correspond to given component?
            if component=='Z':
                _j=0
            elif component=='R':
                _j=1
            elif component=='T':
                _j=2
            G = self._rotated_tensor[_j]

            Mxx =  mt[1]
            Myy =  mt[2]
            Mzz =  mt[0]
            Mxy = -mt[5]
            Mxz = -mt[3]
            Myz =  mt[4]

            # we could use np.dot instead, but any speedup appears negiglibe
            s = self._synthetics[_i].data
            s[:] = 0.
            s += Mxx*G[:,0]
            s += Myy*G[:,1]
            s += Mzz*G[:,2]
            s += Mxy*G[:,3]
            s += Mxz*G[:,4]
            s += Myz*G[:,5]

        return self._synthetics


    def get_time_shift(self, data, mt, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied data
        """
        if not hasattr(self, '_cross_correlation'):
            self._precompute_time_shifts(data, time_shift_max)

        cc = self._cross_correlation
        cc[:] = 0.

        Mxx =  mt[1]
        Myy =  mt[2]
        Mzz =  mt[0]
        Mxy = -mt[5]
        Mxz = -mt[3]
        Myz =  mt[4]

        if 'Z' in self.meta.components:
            CC = self._CCZ
            cc += Mxx*CC[:,0]
            cc += Myy*CC[:,1]
            cc += Mzz*CC[:,2]
            cc += Mxy*CC[:,3]
            cc += Mxz*CC[:,4]
            cc += Myz*CC[:,5]

        if 'R' in self.meta.components:
            CC = self._CCR
            cc += Mxx*CC[:,0]
            cc += Myy*CC[:,1]
            cc += Mzz*CC[:,2]
            cc += Mxy*CC[:,3]
            cc += Mxz*CC[:,4]
            cc += Myz*CC[:,5]

        if 'T' in self.meta.components:
            CC = self._CCT
            cc += Mxx*CC[:,0]
            cc += Myy*CC[:,1]
            cc += Mzz*CC[:,2]
            cc += Mxy*CC[:,3]
            cc += Mxz*CC[:,4]
            cc += Myz*CC[:,5]

        return self._cross_correlation



    def _preallocate_synthetics(self):
        self.meta.npts = self[0].stats.npts
        self._synthetics = Stream()
        for channel in self.meta.components:
            self._synthetics +=\
                Trace(np.zeros(self[0].stats.npts), deepcopy(self.meta))
        self._synthetics.id = self.id


    def _precompute_time_shifts(self, data, max_time_shift):
        """
        Enables fast time-shift calculations by precomputing cross-correlations
        on an element-by-element basis
        """
        npts = self[0].meta['npts']
        npts_padding = int(max_time_shift/self[0].meta['delta'])

        print npts, npts_padding

        self._npts_padding = npts_padding
        self._cross_correlation = np.zeros(2*npts_padding+1)

        if 'Z' in self.meta.components:
            DZ = data.select(component='Z')[0].data
            #DZ = np.pad(DZ, npts_padding, 'constant')

            CCZ = np.zeros((2*npts_padding+1, 6))
            GZ = self._rotated_tensor[0]

        if 'R' in self.meta.components:
            DR = data.select(component='R')[0].data
            #DR = np.pad(DR, npts_padding, 'constant')

            CCR = np.zeros((2*npts_padding+1, 6))
            GR = self._rotated_tensor[1]

        if 'T' in self.meta.components:
            DT = data.select(component='T')[0].data
            #DT = np.pad(DT, npts_padding, 'constant')

            CCT = np.zeros((2*npts_padding+1, 6))
            GT = self._rotated_tensor[2]

        # for long traces or long lag times, frequency-domain
        # implementation is usually faster
        if 'Z' in self.meta.components and\
            (npts > 2000 or npts_padding > 200):
            CCZ[:,0] = fftconvolve(DZ, GZ[::-1,0], 'valid')
            CCZ[:,1] = fftconvolve(DZ, GZ[::-1,1], 'valid')
            CCZ[:,2] = fftconvolve(DZ, GZ[::-1,2], 'valid')
            CCZ[:,3] = fftconvolve(DZ, GZ[::-1,3], 'valid')
            CCZ[:,4] = fftconvolve(DZ, GZ[::-1,4], 'valid')
            CCZ[:,5] = fftconvolve(DZ, GZ[::-1,5], 'valid')
            self._CCZ = CCZ

        if 'R' in self.meta.components and\
            (npts > 2000 or npts_padding > 200):
            CCR[:,0] = fftconvolve(DR, GR[::-1,0], 'valid')
            CCR[:,1] = fftconvolve(DR, GR[::-1,1], 'valid')
            CCR[:,2] = fftconvolve(DR, GR[::-1,2], 'valid')
            CCR[:,3] = fftconvolve(DR, GR[::-1,3], 'valid')
            CCR[:,4] = fftconvolve(DR, GR[::-1,4], 'valid')
            CCR[:,5] = fftconvolve(DR, GR[::-1,5], 'valid')
            self._CCR = CCR

        if 'T' in self.meta.components and\
            (npts > 2000 or npts_padding > 200):

            CCT[:,0] = fftconvolve(DT, GT[::-1,0], 'valid')
            CCT[:,1] = fftconvolve(DT, GT[::-1,1], 'valid')
            CCT[:,2] = fftconvolve(DT, GT[::-1,2], 'valid')
            CCT[:,3] = fftconvolve(DT, GT[::-1,3], 'valid')
            CCT[:,4] = fftconvolve(DT, GT[::-1,4], 'valid')
            CCT[:,5] = fftconvolve(DT, GT[::-1,5], 'valid')
            self._CCT = CCT

        # for short traces or short lag times, time-domain
        # implementation is usually faster
        if 'Z' in self.meta.components and\
            (npts <= 2000 and npts_padding <= 200):
            CCZ[:,0] = np.correlate(DZ, GZ[:,0], 'valid')
            CCZ[:,1] = np.correlate(DZ, GZ[:,1], 'valid')
            CCZ[:,2] = np.correlate(DZ, GZ[:,2], 'valid')
            CCZ[:,3] = np.correlate(DZ, GZ[:,3], 'valid')
            CCZ[:,4] = np.correlate(DZ, GZ[:,4], 'valid')
            CCZ[:,5] = np.correlate(DZ, GZ[:,5], 'valid')
            self._CCZ = CCZ

        if 'R' in self.meta.components and\
            (npts <= 2000 and npts_padding <= 200):
            CCR[:,0] = np.correlate(DR, GR[:,0], 'valid')
            CCR[:,1] = np.correlate(DR, GR[:,1], 'valid')
            CCR[:,2] = np.correlate(DR, GR[:,2], 'valid')
            CCR[:,3] = np.correlate(DR, GR[:,3], 'valid')
            CCR[:,4] = np.correlate(DR, GR[:,4], 'valid')
            CCR[:,5] = np.correlate(DR, GR[:,5], 'valid')
            self._CCR = CCR

        if 'T' in self.meta.components and\
            (npts <= 2000 and npts_padding <= 200):
            CCT[:,0] = np.correlate(DT, GT[:,0], 'valid')
            CCT[:,1] = np.correlate(DT, GT[:,1], 'valid')
            CCT[:,2] = np.correlate(DT, GT[:,2], 'valid')
            CCT[:,3] = np.correlate(DT, GT[:,3], 'valid')
            CCT[:,4] = np.correlate(DT, GT[:,4], 'valid')
            CCT[:,5] = np.correlate(DT, GT[:,5], 'valid')
            self._CCT = CCT



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


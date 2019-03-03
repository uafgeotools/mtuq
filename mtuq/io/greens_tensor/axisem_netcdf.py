
try:
    import instaseis
except:
    pass
import obspy
import numpy as np
import mtuq.io.greens_tensor.base

from os.path import basename
from scipy.signal import fftconvolve
from mtuq.util.signal import resample


# instaseis Green's functions represent vertical, radial, and transverse
# displacement time series (units: m (N-m)^-1)


# instaseis Green's functions describe the impulse response of a horizontally-
# layered medium. For the vertical and raidal components, there are four
# associated time series. For the tranverse component, there are two associated 
# time series. Thus there are ten independent Green's tensor elements altogether, 
# which is fewer than in the case of a general inhomogeneous medium


# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# GreensTensorFactory normally does all the work


class GreensTensor(mtuq.io.greens_tensor.base.GreensTensor):
    """
    AxiSEM Green's tensor object

    Overloads Green's tensor base class with the mathematical machinery
    required for working with AxiSEM-style Green's tensors
    """

    def _precompute_weights(self):
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
        az = np.deg2rad(self.stats.azimuth)

        npts = self[0].stats['npts']
        nc = len(self.components)
        self._rotated_tensor = np.zeros((npts, nc, 6))

        G = self._rotated_tensor
        ic = 0

        if 'Z' in self.components:
            ZSS = self.select(channel="ZSS")[0].data
            ZDS = self.select(channel="ZDS")[0].data
            ZDD = self.select(channel="ZDD")[0].data
            ZEP = self.select(channel="ZEP")[0].data

            G[:, ic, 0] =  ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
            G[:, ic, 1] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
            G[:, ic, 2] =  ZDD/3. + ZEP/3.
            G[:, ic, 3] =  ZSS * np.sin(2*az)
            G[:, ic, 4] =  ZDS * np.cos(az)
            G[:, ic, 5] =  ZDS * np.sin(az)

            ic += 1

        if 'R' in self.components:
            RSS = self.select(channel="RSS")[0].data
            RDS = self.select(channel="RDS")[0].data
            RDD = self.select(channel="RDD")[0].data
            REP = self.select(channel="REP")[0].data

            G[:, ic, 0] =  RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
            G[:, ic, 1] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
            G[:, ic, 2] =  RDD/3. + REP/3.
            G[:, ic, 3] =  RSS * np.sin(2*az)
            G[:, ic, 4] =  RDS * np.cos(az)
            G[:, ic, 5] =  RDS * np.sin(az)

            ic += 1

        if 'T' in self.components:
            TSS = self.select(channel="TSS")[0].data
            TDS = self.select(channel="TDS")[0].data

            G[:, ic, 0] = TSS/2. * np.sin(2*az)
            G[:, ic, 1] = -TSS/2. * np.sin(2*az)
            G[:, ic, 2] = 0.
            G[:, ic, 3] = -TSS * np.cos(2*az)
            G[:, ic, 4] = TDS * np.sin(az)
            G[:, ic, 5] = -TDS * np.cos(az)

            ic += 1


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
        """
        # This moment tensor permutation produces a match between instaseis
        # and fk synthetics.  But what basis conventions does it actually
        # represent?  The permutation appears similar but not identical to the 
        # one that maps from GCMT to AkiRichards
        Mxx =  mt[1]
        Myy =  mt[2]
        Mzz =  mt[0]
        Mxy = -mt[5]
        Mxz = -mt[3]
        Myz =  mt[4]

        if not hasattr(self, '_synthetics'):
            self._preallocate_synthetics()

        if not hasattr(self, '_rotated_tensor'):
            self._precompute_weights()

        for ic, component in enumerate(self.components):
            G = self._rotated_tensor

            # we could use np.dot instead, but speedup appears negligible
            s = self._synthetics[ic].data
            s[:] = 0.
            s += Mxx*G[:, ic, 0]
            s += Myy*G[:, ic, 1]
            s += Mzz*G[:, ic, 2]
            s += Mxy*G[:, ic, 3]
            s += Mxz*G[:, ic, 4]
            s += Myz*G[:, ic, 5]

        return self._synthetics


    def get_time_shift(self, data, mt, group, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied data
        """
        if not hasattr(self, '_cross_correlation'):
            self._precompute_time_shifts(data, time_shift_max)

        cc = self._cross_correlation
        npts_padding = self._npts_padding

        cc[:] = 0.

        # see comments about moment tensor convention in get_synthetics method
        Mxx =  mt[1]
        Myy =  mt[2]
        Mzz =  mt[0]
        Mxy = -mt[5]
        Mxz = -mt[3]
        Myz =  mt[4]

        if 'Z' in group:
            CC = self._CCZ
            cc += Mxx*CC[:,0]
            cc += Myy*CC[:,1]
            cc += Mzz*CC[:,2]
            cc += Mxy*CC[:,3]
            cc += Mxz*CC[:,4]
            cc += Myz*CC[:,5]

        if 'R' in group:
            CC = self._CCR
            cc += Mxx*CC[:,0]
            cc += Myy*CC[:,1]
            cc += Mzz*CC[:,2]
            cc += Mxy*CC[:,3]
            cc += Mxz*CC[:,4]
            cc += Myz*CC[:,5]

        if 'T' in group:
            CC = self._CCT
            cc += Mxx*CC[:,0]
            cc += Myy*CC[:,1]
            cc += Mzz*CC[:,2]
            cc += Mxy*CC[:,3]
            cc += Mxz*CC[:,4]
            cc += Myz*CC[:,5]

        # what is the index of the maximum element of the padded array?
        argmax = cc.argmax()

        # what is the associated lag, in terms of number of samples?
        offset = argmax-npts_padding

        return offset


    def _precompute_time_shifts(self, data, time_shift_max):
        """
        Enables fast time-shift calculations by precomputing cross-correlations
        on an element-by-element basis
        """
        dt = self[0].stats['delta']
        npts = self[0].stats['npts']
        npts_padding = int(time_shift_max/dt)

        self._npts_padding = npts_padding
        self._cross_correlation = np.zeros(2*npts_padding+1)

        if 'Z' in self.components:
            DZ = data.select(component='Z')[0].data
            #DZ = np.pad(DZ, npts_padding, 'constant')

            CCZ = np.zeros((2*npts_padding+1, 6))
            GZ = self._rotated_tensor[0]

        if 'R' in self.components:
            DR = data.select(component='R')[0].data
            #DR = np.pad(DR, npts_padding, 'constant')

            CCR = np.zeros((2*npts_padding+1, 6))
            GR = self._rotated_tensor[1]

        if 'T' in self.components:
            DT = data.select(component='T')[0].data
            #DT = np.pad(DT, npts_padding, 'constant')

            CCT = np.zeros((2*npts_padding+1, 6))
            GT = self._rotated_tensor[2]

        # for long traces or long lag times, frequency-domain
        # implementation is usually faster
        if 'Z' in self.components and\
            (npts > 2000 or npts_padding > 200):
            CCZ[:,0] = fftconvolve(DZ, GZ[::-1,0], 'valid')
            CCZ[:,1] = fftconvolve(DZ, GZ[::-1,1], 'valid')
            CCZ[:,2] = fftconvolve(DZ, GZ[::-1,2], 'valid')
            CCZ[:,3] = fftconvolve(DZ, GZ[::-1,3], 'valid')
            CCZ[:,4] = fftconvolve(DZ, GZ[::-1,4], 'valid')
            CCZ[:,5] = fftconvolve(DZ, GZ[::-1,5], 'valid')
            self._CCZ = CCZ

        if 'R' in self.components and\
            (npts > 2000 or npts_padding > 200):
            CCR[:,0] = fftconvolve(DR, GR[::-1,0], 'valid')
            CCR[:,1] = fftconvolve(DR, GR[::-1,1], 'valid')
            CCR[:,2] = fftconvolve(DR, GR[::-1,2], 'valid')
            CCR[:,3] = fftconvolve(DR, GR[::-1,3], 'valid')
            CCR[:,4] = fftconvolve(DR, GR[::-1,4], 'valid')
            CCR[:,5] = fftconvolve(DR, GR[::-1,5], 'valid')
            self._CCR = CCR

        if 'T' in self.components and\
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
        if 'Z' in self.components and\
            (npts <= 2000 and npts_padding <= 200):
            CCZ[:,0] = np.correlate(DZ, GZ[:,0], 'valid')
            CCZ[:,1] = np.correlate(DZ, GZ[:,1], 'valid')
            CCZ[:,2] = np.correlate(DZ, GZ[:,2], 'valid')
            CCZ[:,3] = np.correlate(DZ, GZ[:,3], 'valid')
            CCZ[:,4] = np.correlate(DZ, GZ[:,4], 'valid')
            CCZ[:,5] = np.correlate(DZ, GZ[:,5], 'valid')
            self._CCZ = CCZ

        if 'R' in self.components and\
            (npts <= 2000 and npts_padding <= 200):
            CCR[:,0] = np.correlate(DR, GR[:,0], 'valid')
            CCR[:,1] = np.correlate(DR, GR[:,1], 'valid')
            CCR[:,2] = np.correlate(DR, GR[:,2], 'valid')
            CCR[:,3] = np.correlate(DR, GR[:,3], 'valid')
            CCR[:,4] = np.correlate(DR, GR[:,4], 'valid')
            CCR[:,5] = np.correlate(DR, GR[:,5], 'valid')
            self._CCR = CCR

        if 'T' in self.components and\
            (npts <= 2000 and npts_padding <= 200):
            CCT[:,0] = np.correlate(DT, GT[:,0], 'valid')
            CCT[:,1] = np.correlate(DT, GT[:,1], 'valid')
            CCT[:,2] = np.correlate(DT, GT[:,2], 'valid')
            CCT[:,3] = np.correlate(DT, GT[:,3], 'valid')
            CCT[:,4] = np.correlate(DT, GT[:,4], 'valid')
            CCT[:,5] = np.correlate(DT, GT[:,5], 'valid')
            self._CCT = CCT




class Client(mtuq.io.greens_tensor.base.Client):
    """ 
    Interface to Instaseis/AxiSEM database of Green's functions

    Generates GreenTensorLists via a two-step procedure
        1) db = mtuq.greens.open_db(path=path, format='instaseis')
        2) greens_tensors = db.read(stations, origin)

    In the first step, the user supplies the path or URL to an AxiSEM NetCDF
    output file

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.

    """

    def __init__(self, path=None, kernelwidth=12):
        try:
            db = instaseis.open_db(path)
        except:
            Exception
        self.db = db
        self.kernelwidth=12


    def _get_greens_tensor(self, station=None, origin=None):
        stream = self.db.get_greens_function(
            epicentral_distance_in_degree=_in_deg(station.distance_in_m),
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

        traces = [trace for trace in stream]
        return GreensTensor(traces=traces, station=station, origin=origin)


def _in_deg(distance_in_m):
    from obspy.geodetics import kilometers2degrees
    return kilometers2degrees(distance_in_m/1000., radius=6371.)


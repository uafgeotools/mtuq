
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


class GreensTensor(mtuq.io.greens_tensor.base.GreensTensor):
    """
    AxiSEM Green's tensor object

    Overloads base class with the mathematical machinery for working with
    AxiSEM-style Green's functions

    AxiSEM Green's functions describe the impulse response of a horizontally-
    layered medium.  Time series represent vertical, radial, and transverse
    displacement in units of m*(N-m)^-1

    For the vertical and raidal components, there are four associated time 
    series. For the tranverse component, there are two associated time
    series. Thus there are ten independent Green's tensor elements altogether, 
    which is fewer than in the case of a general inhomogeneous medium

    For more information, see

    -  van Driel et al. (2015), Instaseis: instant global seismograms,
       Solid Earth, 6, 701-717

    -  Minson, Sarah E. and Dreger, D. (2008), Stable inversions for complete
       moment tensors, GJI, 174 (2): 585-592

    -  github.com/krischer/instaseis/instaseis/tests/
       test_instaseis.py::test_get_greens_vs_get_seismogram

    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        self.tags = []
        self.tags += ['type:greens']
        self.tags += ['type:displacement']
        self.tags += ['units:m']


    def _precompute_weights(self):
        """
        Computes rotated time series used in source-weighted linear 
        combinations
        """
        az = np.deg2rad(self.stats.azimuth)

        # allocate array
        nt = self[0].stats.npts
        nc = len(self.components)
        nr = 9
        G = np.zeros((nc, nr, nt))
        self._rotated_tensor = G

        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = self.select(channel="ZSS")[0].data
                ZDS = self.select(channel="ZDS")[0].data
                ZDD = self.select(channel="ZDD")[0].data
                ZEP = self.select(channel="ZEP")[0].data
                G[_i, 0, :] =  ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
                G[_i, 1, :] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
                G[_i, 2, :] =  ZDD/3. + ZEP/3.
                G[_i, 3, :] =  ZSS * np.sin(2*az)
                G[_i, 4, :] =  ZDS * np.cos(az)
                G[_i, 5, :] =  ZDS * np.sin(az)

            elif component=='R':
                RSS = self.select(channel="RSS")[0].data
                RDS = self.select(channel="RDS")[0].data
                RDD = self.select(channel="RDD")[0].data
                REP = self.select(channel="REP")[0].data
                G[_i, 0, :] =  RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
                G[_i, 1, :] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
                G[_i, 2, :] =  RDD/3. + REP/3.
                G[_i, 3, :] =  RSS * np.sin(2*az)
                G[_i, 4, :] =  RDS * np.cos(az)
                G[_i, 5, :] =  RDS * np.sin(az)

            elif component=='T':
                TSS = self.select(channel="TSS")[0].data
                TDS = self.select(channel="TDS")[0].data
                G[_i, 0, :] = TSS/2. * np.sin(2*az)
                G[_i, 1, :] = -TSS/2. * np.sin(2*az)
                G[_i, 2, :] = 0.
                G[_i, 3, :] = -TSS * np.cos(2*az)
                G[_i, 4, :] = TDS * np.sin(az)
                G[_i, 5, :] = -TDS * np.cos(az)

            else:
                raise ValueError


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

        G = self._rotated_tensor
        for _i, component in enumerate(self.components):

            # we could use np.dot instead, but speedup appears negligible
            s = self._synthetics[_i].data
            s[:] = 0.
            s += Mxx*G[_i, 0, :]
            s += Myy*G[_i, 1, :]
            s += Mzz*G[_i, 2, :]
            s += Mxy*G[_i, 3, :]
            s += Mxz*G[_i, 4, :]
            s += Myz*G[_i, 5, :]

        return self._synthetics


    def get_time_shift(self, data, mt, group, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied data
        """
        if not hasattr(self, '_cc_all'):
            self._precompute_time_shifts(data, time_shift_max)

        npts_padding = self._npts_padding
        cc_all = self._cc_all
        cc_sum = self._cc_sum
        cc_sum[:] = 0.

        # see comments about moment tensor convention in get_synthetics method
        Mxx =  mt[1]
        Myy =  mt[2]
        Mzz =  mt[0]
        Mxy = -mt[5]
        Mxz = -mt[3]
        Myz =  mt[4]

        for component in group:
            _i = self.components.index(component)
            cc_sum += Mxx * cc_all[_i, 0, :]
            cc_sum += Myy * cc_all[_i, 1, :]
            cc_sum += Mzz * cc_all[_i, 2, :]
            cc_sum += Mxy * cc_all[_i, 3, :]
            cc_sum += Mxz * cc_all[_i, 4, :]
            cc_sum += Myz * cc_all[_i, 5, :]

        # what is the index of the maximum element of the padded array?
        argmax = cc_sum.argmax()

        # what is the associated cross correlation lag?
        ioff = argmax-npts_padding

        return ioff


    def _precompute_time_shifts(self, data, time_shift_max):
        """
        Enables fast time-shift calculations by precomputing cross-correlations
        on an element-by-element basis
        """
        dt = self[0].stats['delta']
        npts = self[0].stats['npts']
        npts_padding = int(time_shift_max/dt)
        nc = len(self.components)

        self._npts_padding = npts_padding
        self._cc_sum = np.zeros(2*npts_padding+1)
        self._cc_all = np.zeros((nc, 6, 2*npts_padding+1)) 

        g = self._rotated_tensor
        cc = self._cc_all
        for _i, component in enumerate(self.components):
            d = data.select(component=component)[0].data

            # for long traces or long lag times, frequency-domain
            # implementation is usually faster
            if (npts > 2000 or npts_padding > 200):
                cc[_i, 0, :] = fftconvolve(d, g[_i, 0, ::-1], 'valid')
                cc[_i, 1, :] = fftconvolve(d, g[_i, 1, ::-1], 'valid')
                cc[_i, 2, :] = fftconvolve(d, g[_i, 2, ::-1], 'valid')
                cc[_i, 3, :] = fftconvolve(d, g[_i, 3, ::-1], 'valid')
                cc[_i, 4, :] = fftconvolve(d, g[_i, 4, ::-1], 'valid')
                cc[_i, 5, :] = fftconvolve(d, g[_i, 5, ::-1], 'valid')

            else:
                cc[_i, 0, :] = np.correlate(d, g[_i, 0, :], 'valid')
                cc[_i, 1, :] = np.correlate(d, g[_i, 1, :], 'valid')
                cc[_i, 2, :] = np.correlate(d, g[_i, 2, :], 'valid')
                cc[_i, 3, :] = np.correlate(d, g[_i, 3, :], 'valid')
                cc[_i, 4, :] = np.correlate(d, g[_i, 4, :], 'valid')
                cc[_i, 5, :] = np.correlate(d, g[_i, 5, :], 'valid')



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

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin)


def _in_deg(distance_in_m):
    from obspy.geodetics import kilometers2degrees
    return kilometers2degrees(distance_in_m/1000., radius=6371.)


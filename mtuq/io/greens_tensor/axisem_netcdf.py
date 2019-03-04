
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
    """

    def _precompute_weights(self):
        """
        Calculates weights used in linear combination of Green's functions

        For more information, see

        -  van Driel et al. (2015), Instaseis: instant global seismograms
           Solid Earth, 6, 701-717

        -  Minson, Sarah E. and Dreger, D. (2008), Stable inversions for
           for complete moment tensors, GJI 174 (2): 585-592

        -  github.com/krischer/instaseis/instaseis/tests/
           test_instaseis.py::test_get_greens_vs_get_seismogram
        """
        az = np.deg2rad(self.stats.azimuth)

        npts = self[0].stats['npts']
        nc = len(self.components)
        self._rotated_tensor = {component: np.zeros((6, npts))
            for component in self.components}

        G = self._rotated_tensor

        if 'Z' in self.components:
            ZSS = self.select(channel="ZSS")[0].data
            ZDS = self.select(channel="ZDS")[0].data
            ZDD = self.select(channel="ZDD")[0].data
            ZEP = self.select(channel="ZEP")[0].data

            G['Z'][0, :] =  ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
            G['Z'][1, :] = -ZSS/2. * np.cos(2*az) - ZDD/6. + ZEP/3.
            G['Z'][2, :] =  ZDD/3. + ZEP/3.
            G['Z'][3, :] =  ZSS * np.sin(2*az)
            G['Z'][4, :] =  ZDS * np.cos(az)
            G['Z'][5, :] =  ZDS * np.sin(az)


        if 'R' in self.components:
            RSS = self.select(channel="RSS")[0].data
            RDS = self.select(channel="RDS")[0].data
            RDD = self.select(channel="RDD")[0].data
            REP = self.select(channel="REP")[0].data

            G['R'][0, :] =  RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
            G['R'][1, :] = -RSS/2. * np.cos(2*az) - RDD/6. + REP/3.
            G['R'][2, :] =  RDD/3. + REP/3.
            G['R'][3, :] =  RSS * np.sin(2*az)
            G['R'][4, :] =  RDS * np.cos(az)
            G['R'][5, :] =  RDS * np.sin(az)


        if 'T' in self.components:
            TSS = self.select(channel="TSS")[0].data
            TDS = self.select(channel="TDS")[0].data

            G['T'][0, :] = TSS/2. * np.sin(2*az)
            G['T'][1, :] = -TSS/2. * np.sin(2*az)
            G['T'][2, :] = 0.
            G['T'][3, :] = -TSS * np.cos(2*az)
            G['T'][4, :] = TDS * np.sin(az)
            G['T'][5, :] = -TDS * np.cos(az)


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
            s += Mxx*G[component][0, :]
            s += Myy*G[component][1, :]
            s += Mzz*G[component][2, :]
            s += Mxy*G[component][3, :]
            s += Mxz*G[component][4, :]
            s += Myz*G[component][5, :]

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
            cc_sum += Mxx * cc_all[component][0, :]
            cc_sum += Myy * cc_all[component][1, :]
            cc_sum += Mzz * cc_all[component][2, :]
            cc_sum += Mxy * cc_all[component][3, :]
            cc_sum += Mxz * cc_all[component][4, :]
            cc_sum += Myz * cc_all[component][5, :]

        # what is the index of the maximum element of the padded array?
        argmax = cc_sum.argmax()

        # what is the associated cross correlation lag, in terms of 
        # number of samples?
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

        self._npts_padding = npts_padding
        self._cc_sum = np.zeros(2*npts_padding+1)
        self._cc_all = {component: np.zeros((6, 2*npts_padding+1)) 
                for component in self.components}

        cc = self._cc_all
        for component in self.components:
            d = data.select(component=component)[0].data
            g = self._rotated_tensor[component]

            # for long traces or long lag times, frequency-domain
            # implementation is usually faster
            if (npts > 2000 or npts_padding > 200):
                cc[component][0, :] = fftconvolve(d, g[0, ::-1], 'valid')
                cc[component][1, :] = fftconvolve(d, g[1, ::-1], 'valid')
                cc[component][2, :] = fftconvolve(d, g[2, ::-1], 'valid')
                cc[component][3, :] = fftconvolve(d, g[3, ::-1], 'valid')
                cc[component][4, :] = fftconvolve(d, g[4, ::-1], 'valid')
                cc[component][5, :] = fftconvolve(d, g[5, ::-1], 'valid')

            else:
                cc[component][0, :] = np.correlate(d, g[0, :], 'valid')
                cc[component][1, :] = np.correlate(d, g[1, :], 'valid')
                cc[component][2, :] = np.correlate(d, g[2, :], 'valid')
                cc[component][3, :] = np.correlate(d, g[3, :], 'valid')
                cc[component][4, :] = np.correlate(d, g[4, :], 'valid')
                cc[component][5, :] = np.correlate(d, g[5, :], 'valid')



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


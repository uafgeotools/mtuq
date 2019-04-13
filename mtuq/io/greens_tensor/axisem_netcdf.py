
try:
    import instaseis
except:
    pass
import obspy
import numpy as np

from os.path import basename
from mtuq import GreensTensor as GreensTensorBase
from mtuq.io.greens_tensor.base import Client as ClientBase
from mtuq.util.moment_tensor.basis import change_basis
from mtuq.util.signal import resample
from mtuq.util.util import m_to_deg


class GreensTensor(GreensTensorBase):
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

    See also

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


    def get_synthetics(self, mt):
        """
        Generates synthetics through a source-weighted linear combination
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
        Precomputes time series used in source-weighted linear combinations

        Based on formulas from Minson & Dreger 2008
        """
        phi = np.deg2rad(self.azimuth)

        # array dimensions
        nt = self[0].stats.npts
        nc = len(self.components)
        nr = 6
        if self.enable_force:
            nr += 3

        G = np.zeros((nc, nr, nt))
        self._tensor = G

        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = self.select(channel="ZSS")[0].data
                ZDS = self.select(channel="ZDS")[0].data
                ZDD = self.select(channel="ZDD")[0].data
                ZEP = self.select(channel="ZEP")[0].data
                ZDS *= -1

                G[_i, 0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                G[_i, 1, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                G[_i, 2, :] =  ZDD/3. + ZEP/3.
                G[_i, 3, :] =  ZSS * np.sin(2*phi)
                G[_i, 4, :] =  ZDS * np.cos(phi)
                G[_i, 5, :] =  ZDS * np.sin(phi)

            elif component=='R':
                RSS = self.select(channel="RSS")[0].data
                RDS = self.select(channel="RDS")[0].data
                RDD = self.select(channel="RDD")[0].data
                REP = self.select(channel="REP")[0].data
                RDS *= -1

                G[_i, 0, :] =  RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                G[_i, 1, :] = -RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                G[_i, 2, :] =  RDD/3. + REP/3.
                G[_i, 3, :] =  RSS * np.sin(2*phi)
                G[_i, 4, :] =  RDS * np.cos(phi)
                G[_i, 5, :] =  RDS * np.sin(phi)

            elif component=='T':
                TSS = self.select(channel="TSS")[0].data
                TDS = self.select(channel="TDS")[0].data
                TSS *= -1

                G[_i, 0, :] = TSS/2. * np.sin(2*phi)
                G[_i, 1, :] = -TSS/2. * np.sin(2*phi)
                G[_i, 2, :] = 0.
                G[_i, 3, :] = -TSS * np.cos(2*phi)
                G[_i, 4, :] = TDS * np.sin(phi)
                G[_i, 5, :] = -TDS * np.cos(phi)

            if component=='Z' and\
                self.enable_force:
                Z0 = self.select(channel="Z0")[0].data
                Z1 = self.select(channel="Z1")[0].data
                Z2 = self.select(channel="Z2")[0].data
                G[_i, 6, :] = Z0
                G[_i, 7, :] = Z1
                G[_i, 8, :] = Z2

            elif component=='R' and\
                self.enable_force:
                R0 = self.select(channel="R0")[0].data
                R1 = self.select(channel="R1")[0].data
                R2 = self.select(channel="Z2")[0].data
                G[_i, 6, :] = R0
                G[_i, 7, :] = R1
                G[_i, 8, :] = R2

            elif component=='T' and\
                self.enable_force:
                T0 = self.select(channel="T0")[0].data
                T1 = self.select(channel="T1")[0].data
                T2 = self.select(channel="T2")[0].data
                G[_i, 6, :] = T0
                G[_i, 7, :] = T1
                G[_i, 8, :] = T2



class Client(ClientBase):
    """ 
    Interface to AxiSEM/Instaseis database

    Generates GreenTensorLists via a two-step procedure

    .. code:

        db = mtuq.greens.open_db(path, format='instaseis')

        greens_tensors = db.read(stations, origin)

    In the first step, the user supplies the path or URL to an AxiSEM NetCDF
    output file

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.

    """

    def __init__(self, path_or_url='', kernelwidth=12):
        if not path:
            raise Exception
        try:
            db = instaseis.open_db(path)
        except:
            Exception
        self.db = db
        self.kernelwidth=12


    def _get_greens_tensor(self, station=None, origin=None):
        stream = self.db.get_greens_function(
            epicentral_distance_in_degree=m_to_deg(station.distance_in_m),
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



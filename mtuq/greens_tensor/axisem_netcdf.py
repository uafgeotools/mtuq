
try:
    import instaseis
except:
    pass
import obspy
import numpy as np

from os.path import basename
from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase
from mtuq.util.moment_tensor.basis import change_basis
from mtuq.util.signal import resample
from mtuq.util import m_to_deg



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


    def initialize(self, components):
        """
        Computes numpy arrays used by get_synthetics

        The mathematical formulas below are based on Minson & Dreger 2008
        """
        self.components = components

        for component in components:
            assert component in ['Z', 'R', 'T']


        # allocate obspy stream used by get_synthetics
        self.allocate_synthetics()

        if not components:
            return

        # allocate numpy array used by get_synthetics
        nt = self[0].stats.npts
        nc = len(self.components)
        nr = 6
        if self.enable_force:
            nr += 3
        self._array = np.zeros((nc, nr, nt))


        # fill in the elements of the array
        G = self._array
        phi = np.deg2rad(self.azimuth)
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





import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase
from mtuq.util.moment_tensor.basis import change_basis



class GreensTensor(GreensTensorBase):
    """
    AxiSEM Green's tensor object

    Overloads base class with the mathematical machinery for working with
    AxiSEM-style Green's functions

    .. note:

      AxiSEM Green's functions describe the impulse response of a radially-
      symmetric medium.  Time series represent vertical, radial, and transverse
      displacement in units of m*(N-m)^-1

      For the vertical and raidal components, there are four associated time 
      series. For the tranverse component, there are two associated time
      series. Thus there are ten independent Green's tensor elements altogether,
      which is fewer than in the case of a general inhomogeneous medium

    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        self.tags = []
        self.tags += ['type:greens']
        self.tags += ['type:displacement']
        self.tags += ['units:m']


    def get_synthetics(self, mt):
        """
        Generates synthetics through a linear combination of Green's tensor
        times series weighted by source elements
        """
        return super(GreensTensor, self).get_synthetics(
            change_basis(mt, 1, 2))


    def get_time_shift(self, data, mt, group, time_shift_max):
        """ 
        Finds optimal time shift between the given data and synthetics
        generated from the given source
        """
        return super(GreensTensor, self).get_time_shift(
            data,
            change_basis(mt, 1, 2),
            group,
            time_shift_max)


    def _compute_array(self):
        """
        Computes numpy arrays used by get_synthetics

        The mathematical formulas below are based on Minson & Dreger 2008
        """
        array = self._array
        phi = np.deg2rad(self.azimuth)

        # fill in the elements of the array
        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = self.select(channel="ZSS")[0].data
                ZDS = self.select(channel="ZDS")[0].data
                ZDD = self.select(channel="ZDD")[0].data
                ZEP = self.select(channel="ZEP")[0].data
                ZDS *= -1
                array[_i, 0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, 1, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, 2, :] =  ZDD/3. + ZEP/3.
                array[_i, 3, :] =  ZSS * np.sin(2*phi)
                array[_i, 4, :] =  ZDS * np.cos(phi)
                array[_i, 5, :] =  ZDS * np.sin(phi)

            elif component=='R':
                RSS = self.select(channel="RSS")[0].data
                RDS = self.select(channel="RDS")[0].data
                RDD = self.select(channel="RDD")[0].data
                REP = self.select(channel="REP")[0].data
                RDS *= -1
                array[_i, 0, :] =  RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, 1, :] = -RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, 2, :] =  RDD/3. + REP/3.
                array[_i, 3, :] =  RSS * np.sin(2*phi)
                array[_i, 4, :] =  RDS * np.cos(phi)
                array[_i, 5, :] =  RDS * np.sin(phi)

            elif component=='T':
                TSS = self.select(channel="TSS")[0].data
                TDS = self.select(channel="TDS")[0].data
                TSS *= -1
                array[_i, 0, :] = TSS/2. * np.sin(2*phi)
                array[_i, 1, :] = -TSS/2. * np.sin(2*phi)
                array[_i, 2, :] = 0.
                array[_i, 3, :] = -TSS * np.cos(2*phi)
                array[_i, 4, :] = TDS * np.sin(phi)
                array[_i, 5, :] = -TDS * np.cos(phi)

            if component=='Z' and\
                self.enable_force:
                Z0 = self.select(channel="Z0")[0].data
                Z1 = self.select(channel="Z1")[0].data
                Z2 = self.select(channel="Z2")[0].data
                array[_i, 6, :] = Z0
                array[_i, 7, :] = Z1
                array[_i, 8, :] = Z2

            elif component=='R' and\
                self.enable_force:
                R0 = self.select(channel="R0")[0].data
                R1 = self.select(channel="R1")[0].data
                R2 = self.select(channel="Z2")[0].data
                array[_i, 6, :] = R0
                array[_i, 7, :] = R1
                array[_i, 8, :] = R2

            elif component=='T' and\
                self.enable_force:
                T0 = self.select(channel="T0")[0].data
                T1 = self.select(channel="T1")[0].data
                T2 = self.select(channel="T2")[0].data
                array[_i, 6, :] = T0
                array[_i, 7, :] = T1
                array[_i, 8, :] = T2



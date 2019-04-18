
import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase
from mtuq.util.moment_tensor.basis import change_basis


class GreensTensor(GreensTensorBase):
    """
    FK Green's tensor object

    Overloads base class with the mathematical machinery for working with
    FK-style Green's functions

    .. note:

      FK Green's functions describe the impulse response of a horizontally-
      layered medium.  Time series represent vertical, radial, and transverse
      velocity in units of 10^-20*cm*(dyne-cm)^-1 s^-1

      For the vertical and raidal components, there are four associated time 
      series. For the tranverse component, there are two associated time
      series. Thus there are ten independent Green's tensor elements altogether,
      which is fewer than in the case of a general inhomogeneous medium
    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        self.tags = []
        self.tags += ['type:greens']
        self.tags += ['type:velocity']
        self.tags += ['units:m']


    def get_synthetics(self, mt):
        """
        Generates synthetic seismograms for a given moment tensor, via a linear
        combination of Green's functions
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


    def _compute_array(self, components=['Z', 'R', 'T']):
        array = self._array
        phi = np.deg2rad(self.azimuth)

        # fill in the elements of the array
        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = self.select(channel="ZSS")[0].data
                ZDS = self.select(channel="ZDS")[0].data
                ZDD = self.select(channel="ZDD")[0].data
                ZEP = self.select(channel="ZEP")[0].data
                array[_i, :, 0] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, :, 1] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, :, 2] =  ZDD/3. + ZEP/3.
                array[_i, :, 3] = -ZSS * np.sin(2*phi)
                array[_i, :, 4] = -ZDS * np.cos(phi)
                array[_i, :, 5] = -ZDS * np.sin(phi)

            elif component=='R':
                RSS = self.select(channel="RSS")[0].data
                RDS = self.select(channel="RDS")[0].data
                RDD = self.select(channel="RDD")[0].data
                REP = self.select(channel="REP")[0].data
                array[_i, :, 0] = -RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, :, 1] =  RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, :, 2] =  RDD/3. + REP/3.
                array[_i, :, 3] = -RSS * np.sin(2*phi)
                array[_i, :, 4] = -RDS * np.cos(phi)
                array[_i, :, 5] = -RDS * np.sin(phi)

            elif component=='T':
                TSS = self.select(channel="TSS")[0].data
                TDS = self.select(channel="TDS")[0].data
                array[_i, :, 0] = -TSS/2. * np.sin(2*phi)
                array[_i, :, 1] =  TSS/2. * np.sin(2*phi)
                array[_i, :, 2] =  0.
                array[_i, :, 3] =  TSS * np.cos(2*phi)
                array[_i, :, 4] = -TDS * np.sin(phi)
                array[_i, :, 5] =  TDS * np.cos(phi)

            else:
                raise ValueError



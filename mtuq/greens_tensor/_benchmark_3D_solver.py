
import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase



class GreensTensor(GreensTensorBase):
    """ For the special case where a 3D solver is used with a 1D model
    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        if 'type:greens' not in self.tags:
            self.tags += ['type:greens']

        if 'type:displacement' not in self.tags:
            self.tags += ['type:displacement']

        if 'units:m' not in self.tags:
            self.tags += ['units:m']


    def _precompute(self):
        """ Computes NumPy arrays used by get_synthetics
        """
        if self.include_mt:
            self._precompute_mt()

        if self.include_force:
            self._precompute_force()


    def _precompute_mt(self):
        array = self._array
        phi = np.deg2rad(self.azimuth)
        _j = 0

        for _i, component in enumerate(self.components):
            if component=='Z':
                Grr = self.select(channel="Z.Mrr")[0].data
                Gtt = self.select(channel="Z.Mtt")[0].data
                Gpp = self.select(channel="Z.Mpp")[0].data
                Grt = self.select(channel="Z.Mrt")[0].data

                # convert from up-south-east to north-east-down
                G0 = Gtt
                G1 = Gpp
                G2 = Grr
                G4 = Grt

                # derived by setting phi=0 in Minson2008 formulas
                ZSS = G0 - G1
                ZDS = G4
                ZDD = 2.*G2 - G0 - G1
                ZEP = G0 + G1 + G2

                # Minson2008 formulas
                array[_i, _j+0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, _j+1, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, _j+2, :] =  ZDD/3. + ZEP/3.
                array[_i, _j+3, :] =  ZSS * np.sin(2*phi)
                array[_i, _j+4, :] =  ZDS * np.cos(phi)
                array[_i, _j+5, :] =  ZDS * np.sin(phi)

            elif component=='R':
                Grr = self.select(channel="R.Mrr")[0].data
                Gtt = self.select(channel="R.Mtt")[0].data
                Gpp = self.select(channel="R.Mpp")[0].data
                Grt = self.select(channel="R.Mrp")[0].data

                # convert from up-south-east to north-east-down
                G0 = Gtt
                G1 = Gpp
                G2 = Grr
                G4 = Grt

                # derived by setting phi=0 in Minson2008 formulas
                RSS = G0 - G1
                RDS = G4
                RDD = 2.*G2 - G0 - G1
                REP = G0 + G1 + G2

                # Minson2008 formulas
                array[_i, _j+0, :] =  RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, _j+1, :] = -RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, _j+2, :] =  RDD/3. + REP/3.
                array[_i, _j+3, :] =  RSS * np.sin(2*phi)
                array[_i, _j+4, :] =  RDS * np.cos(phi)
                array[_i, _j+5, :] =  RDS * np.sin(phi)

            elif component=='T':
                Grt = self.select(channel="T.Mrt")[0].data
                Gtp = self.select(channel="T.Mtp")[0].data

                # convert from up-south-east to north-east-down
                G3 = -Gtp
                G5 = -Grt

                TSS = -G3
                TDS = -G5

                array[_i, _j+0, :] = TSS/2. * np.sin(2*phi)
                array[_i, _j+1, :] = -TSS/2. * np.sin(2*phi)
                array[_i, _j+2, :] = 0.
                array[_i, _j+3, :] = -TSS * np.cos(2*phi)
                array[_i, _j+4, :] = TDS * np.sin(phi)
                array[_i, _j+5, :] = -TDS * np.cos(phi)


        #
        # Minson2008 uses a north-east-down basis convention, while mtuq uses an
        # up-south-east basis convention, so a permutation is necessary
        #
        array_copy = array.copy()
        array[:, 0, :] =  array_copy[:, 2, :]
        array[:, 1, :] =  array_copy[:, 0, :]
        array[:, 2, :] =  array_copy[:, 1, :]
        array[:, 3, :] =  array_copy[:, 4, :]
        array[:, 4, :] = -array_copy[:, 5, :]
        array[:, 5, :] = -array_copy[:, 3, :]



    def _precompute_force(self):
        """ Computes NumPy arrays used in force linear combination
        """
        raise NotImplementedError


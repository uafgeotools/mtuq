
import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase



class GreensTensor(GreensTensorBase):
    """
    SPECFEM3D Green's tensor object

    .. note:

      SPECFEM3D Green's functions describe the impulse response of a general
      inhomogeneous medium.  Time series represent vertical, radial, and 
      transverse displacement in units of m*(N-m)^-1 

      (FIXME: DOUBLE CHECK UNITS)

    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        import warnings
        warnings.warn("SPECFEM3D modules have not yet been tested.")

        if 'type:greens' not in self.tags:
            self.tags += ['type:greens']

        if 'type:displacement' not in self.tags:
            self.tags += ['type:displacement']

        if 'units:m' not in self.tags:
            self.tags += ['units:m']




    def _precompute(self):
        """ Computes numpy arrays used by get_synthetics
        """
        if self.include_mt:
            self._precompute_mt()

        if self.include_force:
            self._precompute_force()

        self._permute_array()


    def _precompute_mt(self):
        array = self._array
        phi = np.deg2rad(self.azimuth)
        _j = 0

        for _i, component in enumerate(self.components):
            if component=='Z':
                array[_i, _j+0, :] = self[0].data
                array[_i, _j+1, :] = self[1].data
                array[_i, _j+2, :] = self[2].data
                array[_i, _j+3, :] = self[3].data
                array[_i, _j+4, :] = self[4].data
                array[_i, _j+5, :] = self[5].data

            elif component=='R':
                array[_i, _j+0, :] = self[6].data
                array[_i, _j+1, :] = self[7].data
                array[_i, _j+2, :] = self[8].data
                array[_i, _j+3, :] = self[9].data
                array[_i, _j+4, :] = self[10].data
                array[_i, _j+5, :] = self[11].data

            elif component=='T':
                array[_i, _j+0, :] = self[12].data
                array[_i, _j+1, :] = self[13].data
                array[_i, _j+2, :] = self[14].data
                array[_i, _j+3, :] = self[15].data
                array[_i, _j+4, :] = self[16].data
                array[_i, _j+5, :] = self[17].data


    def _precompute_force(self):
        raise NotImplementedError



    def _permute_array(self):
        # FIXME: DOUBLE CHECK THAT SPECFEM3D WORKS IN UP-SOUTH-EAST
        # CONVENTION
        pass

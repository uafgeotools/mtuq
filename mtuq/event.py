
import numpy as np
import obspy
from obspy.core import UTCDateTime

from mtuq.util import asarray



class Origin(obspy.core.AttribDict):
    """ Origin metadata object

    Holds the following event information

    - latitude and longitude
    - depth
    - origin time

    .. note::

        Some file formats have a corresponding reader that creates
        Origin objects from file metadata (see ``mtuq.io.readers``).

    """
    defaults = {
        'time': UTCDateTime(0),
        'latitude': None,
        'longitude': None,
        'depth_in_m': None,
        }

    def __setitem__(self, key, value):
        if value is None:
            pass

        elif key in ['time']:
            value = UTCDateTime(value)

        elif key in ['latitude', 'longitude', 'depth_in_m']:
            value = float(value)

        super(Origin, self).__setitem__(key, value)


    def __eq__(self, other):
        if not isinstance(other, Origin):
            return False

        for key in self.__dict__:
            if not hasattr(other, key):
                return False
            elif other[key]!=self.__dict__[key]:
                return False
        else:
            return True


class MomentTensor(object):
    """ Moment tensor object

    .. note::
        The constructor accepts moment tensors in any ObsPy `basis convention 
       <https://docs.obspy.org/packages/autogen/obspy.imaging.mopad_wrapper.beach.html#supported-basis-systems>`_.

        In MTUQ code, moment tensors are always internally represented in 
        `'USE'` convention, or in other words ``up-south-east`` convention.

    """
    def __init__(self, array=None, convention="USE"):
        if array is None:
            raise Exception(
               "Missing argument: moment tensors must be given as a "
               "NumPy array")

        try:
            array = asarray(array)
        except:
            raise TypeError(
                "Couldn't cast input argument as NumPy array.")

        if array.size != 6:
            raise ValueError(
                "Unexpected array size: moment tensors must be given as a "
                "NumPy array with six elements")

        if convention.upper() == 'USE':
            self._array = array

        else:
            raise NotImplementedError(
                "So far, only up-south-east convention is implemented")

            #self._array = _change_convention_mt(asarray(array),
            #    asarray(array), convention, 'USE')


    def as_vector(self):
        """ Returns independent elements as 1D NumPy array
        """
        return self._array


    def as_matrix(self):
        """ Returns 2D symmetric NumPy array
        """
        array = self._array
        return np.array([[array[0], array[3], array[4]],
                         [array[3], array[1], array[5]],
                         [array[4], array[5], array[2]]])


    def cast(self, convention):
        """ Returns 1D NumPy array in given basis convention
        """
        return _change_convention_mt(
           self._array, 'USE', convention.upper())


    def moment(self):
        """ Calculates seismic moment (`M_0`)
        """
        M = self.as_matrix()
        return (np.tensordot(M,M)/2.)**0.5


    def magnitude(self):
        """ Calculates moment magnitude (`M_w`)
        """
        # how to generalize for slightly different formulas? 
        return 2./3.*(np.log10(self.moment()) - 9.1)


class Force(object):
    """ Force source

    .. note::
        The constructor accepts forces in any ObsPy `basis convention 
       <https://docs.obspy.org/packages/autogen/obspy.imaging.mopad_wrapper.beach.html#supported-basis-systems>`_.

        In MTUQ code, forces are always internally represented in 
        `'USE'` convention, or in other words ``up-south-east`` convention.

    """
    def __init__(self, array=None, convention="USE"):
        if array is None:
            raise Exception(
               "Missing argument: forces must be given as a "
               "NumPy array")

        try:
            array = asarray(array)
        except:
            raise TypeError(
                "Couldn't cast input argument as NumPy array.")

        if array.size != 3:
            raise ValueError(
                "Unexpected array size: forces must be given as a "
                "NumPy array with three elements")

        if convention.upper() == 'USE':
            self._array = array

        else:
            raise NotImplementedError(
                "So far, only up-south-east convention is implemented")

            #self._array = _change_convention_mt(asarray(array),
            #    asarray(array), convention, 'USE')

    def as_vector(self):
        """ Returns force as NumPy array
        """
        return self._array

    def cast(self, convention):
        """ Returns 1D NumPy array representation in given basis convention
        """
        return _change_convention_force(
           self._array, 'USE', convention.upper())


class CompositeSource(object):
    def __init__(self, sources):
        """ Constructor method
        """
        arrays = []
        size = 0
        for source in sources:
            assert type(source) in [MomentTensor, Force]
            arrays += [source.as_array()]
            size += arrays[-1].size

        self.sources = sources
        self.size = size
        self._array = np.concatenate(arrays)



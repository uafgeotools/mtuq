
import numpy as np
import obspy
from obspy.core import UTCDateTime


class Origin(obspy.core.AttribDict):
    """ Origin metadata object

    Holds the following event information

    - latitude and longitude

    - depth

    - origin time
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
        It is easy to convert from ``MomentTensor`` objects to NumPy arrays 
        using the ``as_vector`` and ``as_matrix`` methods

    """
    def __init__(self, array=None, convention="Unknown"):
        if array is None:
            raise Exception

        if len(array) != 6:
            raise Exception

        try:
            self._array = np.asarray(array)
        except:
            raise TypeError(
                "Couldn't cast input argument as numpy array.")
          
        self.code = convention


    def change_convention(self, code):
        """ Changes basis convention by applying permuation
        """
        if not self.code:
            raise Exception("Can't determine moment tensor permutation "
                "starting basis is unknown.")

        self._arary = _change_convention_mt(
           self._array, self.code, code)


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


    def moment(self):
        """ Calculates seismic moment (`M_0`)
        """
        M = self.as_matrix()
        return (np.tensordot(M,M)/2.)**0.5


    def magnitude(self):
        """ Calculates moment magnitude (`M_w`)
        """
        # how to generalize for users who prefer slightly different formulas? 
        return 2./3.*(np.log10(self.moment()) - 9.1)


class Force(object):
    """ Force source

    .. note::
        It is easy to convert from ``Force`` objects to NumPy arrays using the
        ``as_vector`` method
    """
    def __init__(self, array, convention="Unknown"):
        if array is None:
            raise Exception

        if len(array) != 3:
            raise Exception

        try:
            return np.asarray(array)
        except:
            raise TypeError(
                "Couldn't cast input argument as numpy array.")


    def change_convention(self):
        """ Changes basis convention by applying permuation
        """
        if self.code==0:
            raise Exception

        self._arary = _change_convention_force(self._array, self.code, code)


    def as_vector(self):
        """ Returns force as NumPy array
        """
        return self._array



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



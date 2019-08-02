
import numpy as np
import obspy
from obspy.core import UTCDateTime


class Origin(obspy.core.AttribDict):
    """ Origin metadata object

    Holds the following information
        - hypocenter latitude and longitude
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
        if key in ['time']:
            value = UTCDateTime(value)

        elif key in ['latitude', 'longitude', 'depth_in_m']:
            value = float(value)

        super(Origin, self).__setitem__(key, value)


    def __eq__(self, origin):
        if not isinstance(origin, Origin):
            return False

        for key in self.__dict__:
            if origin[key]!=self.__dict__[key]:
                return False
        else:
            return True


class MomentTensor(object):
    """ Moment tensor object

    .. note:
        For heavy computations, it is sometimes faster to use plain 
        NumPy arrays. It is easy to convert back and forth between
        MomentTensor objects and NumPy arrays using the MomentTensor 
        constructor and the as_vector and as_matrix methods.

    """
    def __init__(self, array=None, code="Unknown"):
        if array is None:
            raise Exception
        if len(array)!=6:
            raise Exception
      
        self._array = _cast(array)
        self.code = _check(code)


    def change_convention(self, code):
        """ Changes basis convention by applying permuation
        """
        if not self.code:
            raise Exception("Can't determine moment tensor permutation "
                "starting basis is unknown.")

        self._arary = _change_convention_mt(
           self._array, self.code, code)


    def as_vector(self):
        """ Returns independent elements as 1D numpy array
        """
        return self._array


    def as_matrix(self):
        """ Returns 2D symmetric numpy array
        """
        array = self._array
        return np.array([[array[0], array[3], array[4]],
                         [array[3], array[1], array[5]],
                         [array[4], array[5], array[2]]])


    def moment(self):
        M = self.as_matrix()
        return (np.tensordot(M,M)/2.)**0.5


    def magnitude(self):
        # how to generalize for users who prefer slightly different formulas? 
        return 2./3.*(np.log10(self.moment()) - 9.1)


class Force(object):
    """ Force source
    """
    def __init__(self, array, convention="Unknown"):
        raise NotImplementedError


    def change_convention(self):
        if self.code==0:
            raise Exception

        self._arary = _change_convention_force(self._array, self.code, code)


    def asvector(self):
        return self._array



#
# utility functions
#

def _cast(array):
    try:
        return np.asarray(array)
    except:
        raise TypeError("Couldn't cast input argument as numpy array.")


def _check(code):
    if code in [0, 1, 2, 3, 4 ,5]:
        return code
    elif code in [0., 1., 2., 3., 4., 5.]:
        return int(code)
    elif code=="Unknown":
        return 0
    elif code=="USE":
        return 1
    elif code=="NED":
        return 2
    elif code=="NWU":
        return 3
    else:
        raise TypeError



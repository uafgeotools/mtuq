
import numpy as np
import obspy
from obspy.core import UTCDateTime


class Origin(obspy.core.AttribDict):
    """ Origin metadata object

    Holds the following information
        - hypocenter location
        - depth
        - origin

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
    """ Moment tensor source
    """
    def __init__(self, array=None, code="Unknown"):
        if not array:
            raise Exception
        if len(array)!=6:
            raise Exception
      
        self._array = _cast(array)
        self.code = _check(code)


    def change_convention(code):
        """ Changes basis convention by applying permuation
        """
        if not self.code:
            raise Exception("Can't determine moment tensor permutation "
                "starting basis is unknown.")

        self._arary = _change_convention_mt(
           self._array, self.code, code)


    def asvector():
        """ Returns independent elements as 1D numpy array
        """
        return self._array


    def asmatrix():
        """ Returns 2D symmetric numpy array
        """
        return np.array([m[0], m[3], m[4]],
                        [m[3], m[1], m[5]],
                        [m[4], m[5], m[2]])


class Force(object):
    """ Force source
    """
    def __init__(self, array, convention="Unknown"):
        raise NotImplementedError


    def change_convention():
        if self.code==0:
            raise Exception

        self._arary = _change_convention_force(self._array, self.code, code)


    def asvector():
        return self._array



class CompositeSource(object):
   """ General combination of force or moment tensor objects
   """
   def __init__(self):
       raise NotImplementedError


#
# utility functions
#

def equals(origin1, origin2):
    for key in origin1:
        if origin1[key] != origin2[key]:
            print key
            return False
    else:
        return True


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
    elif code=="GCMT":
        return 1
    elif code=="AkiRichards":
        return 2
    elif code=="SteinWysession":
        return 3
    else:
        raise TypeError


"""
Basis Convention         Code
======================== ===========================
Unknown                  ``0``, ``"Unknown"``
======================== ===========================
Up-South-East            ``1``, ``"GCMT"``
======================== ===========================
North-East-Down          ``2``, ``"AkiRichards"``
======================== ===========================
North-West-Up            ``3``, ``"SteinWysession"``
======================== ===========================
East-North-Up            ``4`` 
======================== ===========================
South-East-Up            ``5``
======================== ===========================
"""


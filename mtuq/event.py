
import numpy as np
import obspy
from obspy.core import UTCDateTime


class Origin(obspy.core.AttribDict):
    """ Origin metadata object

    Origin encompasses the location, depth, and time of an event

    We use ``obspy.core.traces.Stats`` as a base class because
    ``obspy.core.inventory.station.Station`` is more complex than we need.
    All we really require is an attribute-dictionary.
    """
    defaults = {
        'time': UTCDateTime(0),
        'latitude': None,
        'longitude': None,
        'depth_in_m': None,
        }

    def __setitem__(self, key, value):
        # enforce types
        if key in ['time']:
            value = UTCDateTime(value)

        super(Origin, self).__setitem__(key, value)



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
        """ Returns independent moment tensor elements
        """
        return self._array


    def asmatrix():
        """ Returns symmetric moment tensor matrix
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


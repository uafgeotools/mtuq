
import obspy
import numpy as np

from mtuq.util.geodetics import distance_azimuth
from mtuq.util.signal import convolve


class GreensTensorBase(object):
    """
    Elastic Green's tensor object.  Similar to an obpy Trace, except rather 
    than a single time series, holds multiple time series corresponding to
    the independent elements of an elastic Green's tensor.
    """
    def __init__(self, data, station, origin):
        """
        Normally, all time series required to describe the response at a given
        station to a source at a given origin should be contained in "data".
        Further details regarding how this information is represented are 
        deferred to the subclass, which is necessary because the number of 
        independent Green's tensor elements varies depending on the type medium
        under consideration, among other reasons
        """
        self.data = data
        self.station = station
        self.origin = origin


    def get_synthetics(self, mt):
        """
        Generates synthetic seismogram via linear combination of Green's tensor
        elements
        """
        raise NotImplementedError("Must be implemented by subclass")


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's tensor elements
        """
        raise NotImplementedError("Must be implemented by subclass")


    def convolve(self, wavelet):
        """
        Convolves Green's functions with a source wavelet
        """
        return self.process(convolve, wavelet)
        


class GreensTensorList(object):
    """ 
    A list of GreensTensors.  Similar to an obspy Stream, except rather than 
    traces, holds elastic Green's tensors
    """
    def __init__(self):
        self._list = []


    def __add__(self, greens_tensor):
        self._list += [greens_tensor]
        return self


    def __iter__(self):
        return self._list.__iter__()


    def __getitem__(self, index):
        return self._list[index]


    def __setitem__(self, index, value):
        self._list[index] = value


    @property
    def stations(self):
        stations = []
        for greens_tensor in self._list:
            stations += [greens_tensor.station]
        return stations

    def get_synthetics(self, mt):
        """
        Returns a list of streams; all streams correspond to the same moment
        tensor "mt", and each each individaul stream contains the various
        components recorded a single station
        """
        synthetics = []
        for greens_tensor in self._list:
            synthetics += [greens_tensor.get_synthetics(mt)]
        return synthetics


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's tensors
        """
        processed = GreensTensorList()
        for greens_tensor in self._list:
            processed +=\
                greens_tensor.process(function, *args, **kwargs)
        return processed


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        convolved = GreensTensorList()
        for greens_tensor in self._list:
            convolved += greens_tensor.convolve(wavelet)
        return convolved



class GreensTensorGeneratorBase(object):
    """
    Creates GreensTensorLists via a two-step procedure:

        1) greens_tensor_generator = GreensTensorGenerator(*args, **kwargs)
        2) greens_tensor_list = greens_tensor_generator(stations, origin) 

    In the second step, the user supplies a list of stations and the origin
    location and time information for an event. A GreensTensorList will be
    created containing a GreensTensor for each station-event pair. The order
    of the GreensTensors in the list should match the order of the stations 
    in the input argument.

    Details regarding how the GreenTensors are actually created--whether
    they are computed on-the-fly or read from a pre-computed database--
    are deferred to the subclass.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


    def __call__(self, stations, origin, verbose=False):
        """
        Reads Green's tensors corresponding to given stations and origin
        """
        greens_tensor_list = GreensTensorList()

        for station in stations:
            # add distance and azimuth to station metadata
            station.distance, station.azimuth = distance_azimuth(
                station, origin)

            # add another GreensTensor to list
            greens_tensor_list += self.get_greens_tensor(
                station, origin)

        return greens_tensor_list


    def get_greens_tensor(self, station, origin):
        raise NotImplementedError("Must be implemented by subclass")


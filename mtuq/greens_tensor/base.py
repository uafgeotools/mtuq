
import obspy
import numpy as np

from mtuq.util.geodetics import distance_azimuth
from mtuq.util.signal import convolve


class GreensTensorBase(object):
    """ Elastic Green's tensor object.  

        Similar to an obpy Trace, except rather than a single time series, holds
        multiple time series corresponding to the independent elements of an 
        elastic Green's tensor.
    """

    def __init__(self, data, station, origin):
        """
        Normally, all time series required to describe the response at a given
        station to a source at a given origin should be contained in "data".
        Further details regarding how this information is represented are 
        deferred to the subclass
        """
        self.data = data
        self.station = station
        self.origin = origin

        self.assign_id()


    def assign_id(self):
        """
        Assigns a unique identifier, in this case based on network and station
        names, but can be modified by subclass
        """
        self.id = '.'.join((
            self.station.network,
            self.station.station,
            self.station.location))


    def get_synthetics(self, mt):
        """
        Generates synthetic seismogram via linear combination of Green's tensor
        elements
        """
        raise NotImplementedError("Must be implemented by subclass")


    def apply(self, function, *args, **kwargs):
        """
        Applies a function to all time series associated with the given 
        Green's tensor
        """
        raise NotImplementedError("Must be implemented by subclass")


    def convolve(self, wavelet):
        """
        Convolves source wavelet with all time series associated with the
        given Green's tensor
        """
        return self.apply(wavelet.convolve_stream)
        


class GreensTensorList(object):
    """ A list of GreensTensors

        Very similar to an MTUQ Dataset, except rather observed data, holds
        synthetic Green's tensors
    """
    def __init__(self):
        self.__list__ = []


    def get_synthetics(self, mt):
        """
        Returns an MTUQ Dataset in which all streams correspond to the moment
        tensor mt, and each each individaul stream corresponds to an
        individual station
        """
        synthetics = []
        for greens_tensor in self.__list__:
            synthetics += [greens_tensor.get_synthetics(mt)]
        return synthetics


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        convolved = GreensTensorList()
        for greens_tensor in self.__list__:
            convolved += greens_tensor.convolve(wavelet)
        return convolved


    def apply(self, function, *args, **kwargs):
        """
        Returns the result of applying a function to each GreensTensor in the 
        list. Similar to the behavior of the python built-in "apply".
        """
        processed = GreensTensorList()
        for greens_tensor in self.__list__:
            processed +=\
                greens_tensor.apply(function, *args, **kwargs)
        return processed


    def map(self, function, *sequences):
        """
        Returns the result of applying a function to each GreensTensor in the
        list. If one or more optional sequences are given, the function is 
        called with an argument list consisting of the corresponding item of
        each sequence. Similar to the behavior of the python built-in "map".
        """
        processed = GreensTensorList()
        for _i, greens_tensor in enumerate(self.__list__):
            args = [sequence[_i] for sequence in sequences]
            processed +=\
                greens_tensor.apply(function, *args)
        return processed


    # the remaining methods deal with indexing and iteration over the 
    # the list of GreensTensors
    def __add__(self, greens_tensor):
        #assert hasattr(greens_tensor, 'id')
        greens_tensor.tag = 'greens_tensor'
        self.__list__ += [greens_tensor]
        return self


    def remove(self, id):
        index = self._get_index[id]
        self.__list__.pop(index)


    def _get_index(self, id):
        for index, greens_tensor in enumerate(self.__list__):
            if id==greens_tensor.id:
                return index


    def __iter__(self):
        return self.__list__.__iter__()


    def __getitem__(self, index):
        return self.__list__[index]


    def __setitem__(self, index, value):
        self.__list__[index] = value


    def __len__(self):
        return len(self.__list__)



class GeneratorBase(object):
    """
    Creates GreensTensorLists via a two-step procedure:

        1) greens_tensor_generator = GreensTensorGenerator(*args, **kwargs)
        2) greens_tensors = greens_tensor_generator(stations, origin) 

    In the second step, the user supplies a list of stations and the origin
    location and time information for an event. A GreensTensorList will be
    created containing a GreensTensor for each station-event pair. The order
    of the GreensTensors in the list should match the order of the stations 
    in the input argument.

    Details regarding how the GreenTensors are actually created--whether
    they are computed on-the-fly or read from a pre-computed database--
    are deferred to the subclass.

    Very similar to an mtuq.dataset.reader, excecpt rather than a Dataset,
    returns a GreensTensorsList
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


    def __call__(self, stations, origin, verbose=False):
        """
        Reads Green's tensors corresponding to given stations and origin
        """
        greens_tensors = GreensTensorList()

        for station in stations:
            # add distance and azimuth to station metadata
            station.distance, station.azimuth = distance_azimuth(
                station, origin)

            # add another GreensTensor to list
            greens_tensors += self.get_greens_tensor(
                station, origin)

        return greens_tensors


    def get_greens_tensor(self, station, origin):
        raise NotImplementedError("Must be implemented by subclass")


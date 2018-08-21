
import numpy as np

from copy import deepcopy
from obspy.core import Stream, Trace
from scipy.signal import fftconvolve
from mtuq.dataset.base import Dataset
from mtuq.util.geodetics import distance_azimuth
from mtuq.util.signal import check_time_sampling, convolve
from mtuq.util.util import iterable


class GreensTensor(Stream):
    """ Elastic Green's tensor object

        Holds multiple time series corresponding to the independent elements 
        of an elastic Green's tensor. Besides data processing utilities
        inherited from obspy.Stream, provides methods for generating 
        synthetics, convolving source wavelets, and calculating 
        cross-correlation time-shifts.
    """

    def __init__(self, traces, station, origin):
        assert check_time_sampling(traces), NotImplementedError(
            "Time sampling differs from trace to trace.")

        super(GreensTensor, self).__init__(traces)

        self.id = station.id
        self.tags = ['greens_tensor']
        self.meta = deepcopy(station)
        self.origin = origin

        # when generating synthetics, create these components
        self.components = ['Z', 'R', 'T']


    def get_synthetics(self, mt):
        """
        Generates synthetics through a linear combination of time series
        """
        raise NotImplementedError("Must be implemented by subclass")


    def _precompute_weights(self):
        """
        Calculates weights used in linear combination of Green's functions
        """
        raise NotImplementedError("Must be implemented by subclass")


    def _preallocate_synthetics(self):
        """
        Enables fast synthetics calculations through preallocation and
        and memory reuse
        """
        self._synthetics = Stream()
        for channel in self.components:
            meta = deepcopy(self.meta)
            meta.update({
                'npts': self[0].stats.npts,
                'channel': channel,
                })
            self._synthetics += Trace(np.zeros(meta.npts), meta)

        self._synthetics.id = self.id


    def apply(self, function, *args, **kwargs):
        """
        Applies a function to all time series
        """
        return self.__class__(function(self, *args, **kwargs),
            self.meta, self.origin)


    def convolve(self, wavelet):
        """
        Convolves source wavelet with all time series
        """
        return self.apply(wavelet.convolve_stream)


    def get_time_shift(self, data, mt, group, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied data
        """
        raise NotImplementedError("Must be implemented by subclass")


    def _precompute_time_shifts(self, data, max_time_shift):
        """
        Enables fast time-shift calculations by computing cross-correlations
        on an element-by-element basis
        """
        raise NotImplementedError("Must be implemented by subclass")


    def select(self, *args, **kwargs):
        """
        Same as obspy.Stream.select
        """
        # convert to Stream
        stream = Stream([trace for trace in self]).select(*args, **kwargs)
        
        # convert back to GreensTensor
        return self.__class__(
            [trace for trace in stream],
            self.meta,
            self.origin)


    def __add__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")


    def __iadd__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")



class GreensTensorList(object):
    """ A list of GreensTensors

        Very similar to an MTUQ Dataset, except rather observed data, holds
        synthetic Green's tensors
    """
    def __init__(self, greens_tensors=None, id=None):
        # typically the id is the event name, event origin time, or some other
        # attribute shared by all GreensTensors
        self.id = id
        self.__list__ = []

        if not greens_tensors:
            # return an empty container, GreensTensors can be added later
            return

        for greens_tensor in greens_tensors:
            self.__add__(greens_tensor)


    def get_synthetics(self, mt):
        """
        Returns an MTUQ Dataset in which all streams correspond to the moment
        tensor mt, and each each individual stream corresponds to an
        individual station
        """
        synthetics = Dataset()
        for greens_tensor in self.__list__:
            synthetics += greens_tensor.get_synthetics(mt)
        return synthetics


    # the next three methods can be used to apply signal processing or other
    # operations to all time series in all GreensTensors
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
        Applies a function in-pace to each GreensTensor in the list. If one or
        more optional sequences are given, the function is called with an 
        argument list consisting of the corresponding item of each sequence. 
        Similar to the behavior of the python built-in "map".
        """
        processed = GreensTensorList()
        for _i, greens_tensor in enumerate(self.__list__):
            args = [sequence[_i] for sequence in sequences]
            processed +=\
                greens_tensor.apply(function, *args)
        return processed


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        convolved = GreensTensorList()
        for greens_tensor in self.__list__:
            convolved += greens_tensor.convolve(wavelet)
        return convolved


    def add_tag(self, tag):
       for greens_tensor in self:
           greens_tensor.tags.append(tag)


    def remove_tag(self, tag):
       for greens_tensor in self:
           greens_tensor.tags.remove(tag)


    # the next method is called repeatedly during GreensTensorList creation
    def __add__(self, greens_tensor):
        #assert hasattr(greens_tensor, 'id')
        self.__list__ += [greens_tensor]
        return self


    def remove(self, id):
        index = self._get_index(id)
        self.__list__.pop(index)


    # various sorting methods
    def sort_by_distance(self, reverse=False):
        """ 
        Sorts in-place by hypocentral distance
        """
        self.sort_by_function(lambda stream: stream.meta.distance,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """
        Sorts in-place by source-receiver azimuth
        """
        self.sort_by_function(lambda stream: stream.meta.azimuth,
            reverse=reverse)


    def sort_by_function(self, function, reverse=False):
        """ 
        Sorts in-place using the python built-in "sort"
        """
        self.__list__.sort(key=function, reverse=reverse)


    # the remaining methods deal with indexing and iteration
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



class GreensTensorFactory(object):
    """
    Creates GreensTensorLists via a two-step procedure:

        1) factory = GreensTensorFactory(*args, **kwargs)
        2) greens_tensors = factory(stations, origin) 

    In the second step, the user supplies a list of station objects and
    an origin object. The result is a GreensTensorList containing a
    GreensTensor for each station-event pair, with the order of GreensTensors 
    matching the order of the stations in the input argument.

    Details regarding how the GreenTensors are actually created--whether
    they are generated on-the-fly or read from a pre-computed database--
    are deferred to the subclass.

    This class's __call__ method is very similar to an mtuq.dataset.reader, 
    excecpt it returns a GreensTensorsList rather than a Dataset
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")


    def __call__(self, stations, origin, verbose=False):
        """
        Reads Green's tensors corresponding to given stations and origin
        """
        greens_tensors = GreensTensorList()

        for station in iterable(stations):
            # if hypocenter is an inversion parameter, then the values 
            # calculated below will in general differ from catalog_distance and
            # catalog_azimuth
            station.distance, station.azimuth = distance_azimuth(
                station, origin)

            greens_tensors += self.get_greens_tensor(
                station, origin)

        return greens_tensors


    def get_greens_tensor(self, station, origin):
        raise NotImplementedError("Must be implemented by subclass")


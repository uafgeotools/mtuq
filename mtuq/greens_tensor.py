
import numpy as np

from copy import deepcopy
from obspy.core import Stream, Trace
from mtuq.dataset import Dataset
from mtuq.util.signal import check_time_sampling


class GreensTensor(Stream):
    """ Green's tensor base class

        Holds multiple time series corresponding to the independent elements 
        of an elastic Green's tensor. 

        .. note::

            Besides those methods described below, also includes data 
            processing methods inherited from ``obspy.Stream``.
            For descriptions of inherited methods, see `ObsPy documentation
            <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.htm>`_
    """

    def __init__(self, traces, station, origin):
        assert check_time_sampling(traces), NotImplementedError(
            "Time sampling differs from trace to trace.")

        super(GreensTensor, self).__init__(traces)

        self.id = station.id
        self.tags = ['type:greens_tensor']
        self.stats = deepcopy(station)
        self.origin = origin


    def get_synthetics(self, mt):
        """
        Generates synthetics through a linear combination of Green's tensor
        elements
        """
        raise NotImplementedError("Must be implemented by subclass")


    def get_time_shift(self, data, mt, group, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied data

        :type data: mtuq.Dataset
        :param data: Data to be cross-correlated with synthetics
        :type mt: mtuq.MomentTensor
        :param mt: Moment tensor used to generate synthetics
        :type group: str
        :param group: Which components to consider when computing optimal time-
            shift, e.g. ``"Z"``, ``"R"``, ``"T"``, ``"ZR"``, ``"ZRT"``. If 
            multiple characters are given, then time-shift will be fixed across
            multiple components.
        :type time_shift_max: float
        :param time_shift_max: Maximum allowable time-shift. Lag times greater 
            than this value are not computed in the cross-correlation.
        """
        raise NotImplementedError("Must be implemented by subclass")


    def apply(self, function, *args, **kwargs):
        """ 
        Applies a function to all time series

        :type function: func
        :param function: Any function that acts on an obspy ``Stream``
        :rtype: Always returns a new ``GreensTensor``. (Never operates in-place
           on the existing one.)
        
        """
        return self.__class__(function(self, *args, **kwargs),
            self.stats, self.origin)


    def convolve(self, wavelet):
        """
        Convolves source wavelet with all time series

        :type wavelet: mtuq.util.wavelets.Wavelet
        :param wavelet: Source wavelet or source-time function to be used in
            convolution
        """
        for trace in self:
            wavelet.convolve(trace)


    def _preallocate_synthetics(self):
        """
        Enables fast synthetics calculations through preallocation and
        and memory reuse
        """
        self._synthetics = Stream()
        for channel in self.components:
            stats = deepcopy(self.stats)
            stats.update({
                'npts': self[0].stats.npts,
                'channel': channel,
                })
            self._synthetics += Trace(np.zeros(stats.npts), stats)

        self._synthetics.id = self.id


    def _precompute_weights(self):
        """
        Calculates weights used in linear combination of Green's functions
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
        Return new GreensTensor with only those traces that match the supplied
        metadata criteria
        """
        # Inherited method "select" doesn't work because it tries to return a
        # a parent class instance. To fix it, we need to convert to Stream,
        # then call the method, then convert back to  GreensTensor

        stream = Stream([trace for trace in self]).select(*args, **kwargs)

        return self.__class__(
            [trace for trace in stream],
            self.stats,
            self.origin)


    def __add__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")


    def __iadd__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")



class GreensTensorList(object):
    """ Container for one or more GreensTensor objects

    Basically, a list of GreensTensors. Very similar to an MTUQ Dataset, except 
    rather than a list of Streams containing  observed data, holds synthetic 
    Green's tensors
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
        """ Generates synthetic seismograms by summing Green's functions in a 
        linear combination weighted by moment tensor elements

        :param mt: Moment tensor to be used in linear combination
        :rtype: An MTUQ ``Dataset`` in which all streams correspond to the
            given moment tensor, and each each individual stream corresponds to
            an individual station
        """
        synthetics = Dataset()
        for greens_tensor in self.__list__:
            synthetics += greens_tensor.get_synthetics(mt)
        return synthetics


    # the next three methods can be used to apply signal processing or other
    # operations to all time series in all GreensTensors
    def apply(self, function, *args, **kwargs):
        """ Returns the result of applying a function to each GreensTensor in
        the list. Similar to the Python built-in "apply".

        :type function: func
        :param function: Function that acts on obspy an ``Stream``
        :rtype: ``GreensTensorList``
        """
        processed = GreensTensorList()
        for greens_tensor in self.__list__:
            processed +=\
                greens_tensor.apply(function, *args, **kwargs)
        return processed


    def map(self, function, *sequences):
        """ Applies a function in-pace to each GreensTensor in the list. If one
        or more optional sequences are given, the function is called with an 
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
        self.sort_by_function(lambda stream: stream.stats.distance,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """
        Sorts in-place by source-receiver azimuth
        """
        self.sort_by_function(lambda stream: stream.stats.azimuth,
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





import numpy as np

from copy import copy
from mtuq.event import Origin
from mtuq.station import Station
from mtuq.dataset import Dataset
from mtuq.util.signal import check_time_sampling
from obspy.core import Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import fftconvolve


class GreensTensor(Stream):
    """
    Holds multiple time series corresponding to the independent elements 
    of an elastic Green's tensor. 

    .. note::

        Besides the methods described below, GreensTensor also includes
        data processing methods inherited from ``obspy.core.Stream``.
        For descriptions of inherited methods, see `ObsPy documentation
        <https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.htm>`_
    """
    def __init__(self, 
            traces=None, 
            station=None, 
            origin=None,
            id=None, 
            tags=[],
            include_mt=True,
            include_force=False):
        """ Constructor method
        """

        # argument checking starts now
        for trace in traces:
            assert isinstance(trace, Trace)

        assert check_time_sampling(traces), NotImplementedError(
            "Time sampling differs from trace to trace.")

        super(GreensTensor, self).__init__(traces)

        if not isinstance(station, Station):
            raise TypeError

        if not isinstance(origin, Origin):
            raise TypeError

        # the main work of the constructor starts now
        if id:
            self.id = id
        else:
            self.id = station.id

        self.station = station.copy()
        self.origin = origin.copy()
        self.tags = tags
        self.include_mt = include_mt
        self.include_force = include_force

        self.distance_in_m, self.azimuth, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)


    def _set_components(self, components):
        """
        This method must be called prior to ``get_synthetics`` to specify which
        components are returned

        .. note:

           Sometimes it makes sense to return no components at a particular
           station, for example, if all components are absent from the recorded
           data.

        .. note:

          Sometimes it makes sense to call this method partway through a 
          script. For example, if the transerve component at a certain station
          is discovered to be bad, calling ``_set_components(['Z', 'R'])``
          would remove the transverse component, avoiding unnecessary 
          computations.

        """
        if components==getattr(self, 'components', None):
            return

        if components is None:
            components = []

        for component in components:
            assert component in ['Z', 'R', 'T']

        self.components = components

        self._preallocate()
        self._precompute()


    def _preallocate(self):
        """
        Preallocates structures used by ``get_synthetics``

        .. note:

            Every time ``get_synthetics`` is called, the numeric trace data 
            gets overwritten. Every time ``_set_components`` is called, the 
            traces get overwritten.  The stream itself never gets overwritten.
        """
        # allocate array to hold linear combination time series
        nt = len(self[0].data)
        nc = len(self.components)
        nr = 0
        if self.include_mt:
            nr += 6
        if self.include_force:
            nr+= 3
        self._array = np.zeros((nc, nr, nt))

        # allocate obspy structures to hold synthetics
        self._synthetics = Stream()
        self._synthetics.station = self.station
        self._synthetics.origin = self.origin
        for component in self.components:
            # add stats object
            stats = self.station.copy()
            stats.update({'npts': nt, 'channel': component})
            # add trace object
            self._synthetics += Trace(np.zeros(nt), stats)


    def _precompute(self):
        """
        Precomputes numpy array used by ``get_synthetics``
        """
        # the formulas relating the original time series to the linear
        # combination array vary depending on the scheme being used, so
        # are deferred to the subclass
        raise NotImplementedError("Must be implemented by subclass.")


    def get_synthetics(self, source, components=None):
        """
        Generates synthetics through a linear combination of time series
        """

        if components is None:
            # Components argument was not given, so check that attribute is
            # already set
            assert(hasattr(self, 'components'))

        else:
            self._set_components(components)

        # arrays used in linear combination
        array = self._array
        synthetics = self._synthetics

        for _i, component in enumerate(self.components):
            # Even with careful attention to index order, np.dot is very slow.
            # For some reason the following is faster
            data = synthetics[_i].data
            data[:] = 0.
            for _j in range(len(source)):
                data += source[_j]*array[_i, _j, :]
        return synthetics


    def convolve(self, wavelet):
        """
        Convolves time series with the given wavelet

        :type wavelet: mtuq.util.wavelets.Wavelet
        :param wavelet: Source wavelet to be convolved
        """
        for trace in self:
            wavelet.convolve(trace)


    def select(self, component=None, channel=None):
        """
        Selects time series that match the supplied metadata criteria
        """
        return Stream([trace for trace in self]).select(
            component=component, channel=channel)


    def __add__(self, *args):
        raise Exception("Adding time series to an existing GreensTensor is "
           " not currently supported")


    def __iadd__(self, *args):
        raise Exception("Adding time series to an existing GreensTensor is "
           " not currently supported")



class GreensTensorList(list):
    """ Container for one or more GreensTensor objects
    """

    def __init__(self, tensors=[], id=None, tags=[]):
        # typically the id is the event name or origin time
        self.id = id

        for tensor in tensors:
            self.append(tensor)

        for tag in copy(tags):
            self.tag_add(tag)


    def append(self, tensor):
        """ Appends GreensTensor to list
        """
        if not hasattr(tensor, 'station'):
            raise Exception("GreensTensor lacks station metadata")

        elif not hasattr(tensor, 'origin'):
            raise Exception("GreensTensor lacks origin metadata")

        super(GreensTensorList, self).append(tensor)


    def select(self, selector):
        """ Selects GreensTensors that match the given station or origin
        """
        if type(selector) is Station:
            return self.__class__(id=self.id, tensors=filter(
                lambda tensor: tensor.station==selector, self))

        elif type(selector) is Origin:
            return self.__class__(id=self.id, tensors=filter(
                lambda tensor: tensor.origin==selector, self))



    def get_synthetics(self, source, **kwargs):
        """ Generates synthetic by linear combination of Green's functions 
        """
        synthetics = Dataset()
        for tensor in self:
            synthetics.append(tensor.get_synthetics(source, **kwargs))
        return synthetics


    # the next three methods can be used to apply signal processing or other
    # operations to all time series in all GreensTensors
    def apply(self, function, *args, **kwargs):
        """ Applies function to all GreensTensors
 
        Applies a function to each GreensTensor in the list, identical to the 
        Python built-in ``apply``.  

        .. warning ::

            Although ``map`` returns a new GreensTensorList, it is possible,
            depending on the behavior of the given function, that the elements 
            of the original list are overwritten.

            See also ``mtuq.process_data.ProcessData``, which has an
            `overwrite` keyword argument that is `False` by default.

        """
        processed = []
        for tensor in self:
            processed +=\
                [function(tensor, *args, **kwargs)]
        return self.__class__(processed)


    def map(self, function, *sequences):
        """ Maps function to all GreensTensors

        Maps a function to each GreensTensor in the list. If one or more 
        optional sequences are given, the function is called with an argument 
        list consisting of the corresponding item of each sequence, identical 
        to the Python built-in ``map``.

        .. warning ::

            Although ``map`` returns a new GreensTensorList, it is possible,
            depending on the behavior of the given function, that the elements 
            of the original list are overwritten.

            See also ``mtuq.process_data.ProcessData``, which has an
            `overwrite` keyword argument that is `False` by default.

        """
        processed = []
        for _i, tensor in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed +=\
                [function(tensor, *args)]
        return self.__class__(processed)


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        for tensor in self:
            tensor.convolve(wavelet)


    def tag_add(self, tag):
       """ Appends string to tags list
       
       Tags can be used to support customized uses, such as storing metdata not
       included in ``mtuq.Station``
       """
       if type(tag) not in [str, unicode]:
           raise TypeError

       for tensor in self:
           if tag not in tensor.tags:
               tensor.tags.append(tag)


    def tag_remove(self, tag):
       """ Removes string from tags list
       """
       for tensor in self:
           if tag in tensor.tags:
               tensor.tags.remove(tag)


    def sort_by_distance(self, reverse=False):
        """ 
        Sorts in-place by hypocentral distance
        """
        self.sort_by_function(lambda stream: stream.distance,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """
        Sorts in-place by source-receiver azimuth
        """
        self.sort_by_function(lambda stream: stream.azimuth,
            reverse=reverse)


    def sort_by_function(self, function, reverse=False):
        """ 
        Sorts in-place using the python built-in "sort"
        """
        self.sort(key=function, reverse=reverse)



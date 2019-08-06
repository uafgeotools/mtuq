
import numpy as np

from mtuq.event import Origin
from mtuq.station import Station
from mtuq.dataset import Dataset
from mtuq.util.signal import check_time_sampling
from obspy.core import Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import fftconvolve


class GreensTensor(Stream):
    """
    Green's tensor base class

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
            model=None,
            solver=None,
            components=[],
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

        if components==None:
            components = []
        for component in components:
            assert component in ['Z', 'R', 'T']

        # the main work of the constructor starts now
        if id:
            self.id = id
        else:
            self.id = station.id

        self.station = station.copy()
        self.origin = origin.copy()
        self.model = model
        self.solver = solver
        self.include_mt = include_mt
        self.include_force = include_force
        self.components = components

        self.distance_in_m, self.azimuth, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        self._preallocate()
        self._precompute()


    def reset_components(self, components):
        """
        Resets components returned by ``get_synthetics``

        Suppose the vertical and radial components of the recorded data are
        good but the transerve component is found to be bad.  Calling
        ``reset_components(['Z', 'R'])`` will cause ``get_synthetics`` to only
        return the two good components, avoiding unnecessary computation of
        the bad component
        """
        if components==self.components:
            return

        elif components==None:
            components = []

        for component in components:
            assert component in ['Z', 'R', 'T']

        self.components = components

        self._preallocate()
        self._precompute()


    def _preallocate(self):
        """
        Preallocates structures used by ``get_synthetics``

        Every time ``get_synthetics`` is called, the numeric trace data gets
        overwritten. Every time ``reset_components`` is called, the traces
        themselves get overwritten.  The stream itself never gets overwritten.
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
        if not hasattr(self, '_synthetics'):
            self._synthetics = Stream()
            self._synthetics.station = self.station
            self._synthetics.origin = self.origin
        for trace in self._synthetics:
            self._synthetics.remove(trace)
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


    def get_synthetics(self, source):
        """
        Generates synthetics through a linear combination of time series
        """
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


    def apply(self, function, *args, **kwargs):
        """ 
        Applies a function to all time series

        :type function: func
        :param function: Any function that acts on an obspy ``Stream``
        :rtype: Always returns a new ``GreensTensor``. (Never operates in-place
           on the existing one.)
        
        """
        return self.__class__(function(self, *args, **kwargs),
            station=self.station,
            origin=self.origin,
            id=self.id,
            model=self.model,
            solver=self.solver,
            include_mt=self.include_mt,
            include_force=self.include_force)


    def convolve(self, wavelet):
        """
        Convolves ``wavelet`` with all time series

        :type wavelet: mtuq.util.wavelets.Wavelet
        :param wavelet: Source wavelet or source-time function to be used in
            convolution
        """
        for trace in self:
            wavelet.convolve(trace)


    def select(self, component=None, channel=None):
        """
        Return stream with only those traces that match the supplied
        metadata criteria
        """
        return Stream([trace for trace in self]).select(
            component=component, channel=channel)


    def __add__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")


    def __iadd__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")



class GreensTensorList(list):
    """ Container for one or more GreensTensor objects
    """

    def __init__(self, tensors=[], id=None):
        # typically the id is the event name or origin time
        self.id = id

        for tensor in tensors:
            self.append(tensor)


    def append(self, tensor):
        """ Appends GreensTensor to list
        """
        if not hasattr(tensor, 'station'):
            raise Exception("GreensTensor lacks station metadata")

        elif not hasattr(tensor, 'origin'):
            raise Exception("GreensTensor lacks origin metadata")

        super(GreensTensorList, self).append(tensor)


    def select(self, origin):
        """ Selects GreensTensors that match the given origin
        """
        return self.__class__(id=self.id, tensors=filter(
            lambda tensor: tensor.origin==origin, self))


    def get_synthetics(self, source):
        """ Generates synthetic by linear combination of Green's functions 
        """
        synthetics = Dataset()
        for tensor in self:
            synthetics.append(tensor.get_synthetics(source))
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
        processed = []
        for tensor in self:
            processed +=\
                [tensor.apply(function, *args, **kwargs)]
        return self.__class__(processed)


    def map(self, function, *sequences):
        """ Applies a function in-pace to each GreensTensor in the list. If one
        or more optional sequences are given, the function is called with an 
        argument list consisting of the corresponding item of each sequence. 
        Similar to the behavior of the python built-in "map".
        """
        processed = []
        for _i, tensor in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed +=\
                [tensor.apply(function, *args)]
        return self.__class__(processed)


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        for tensor in self:
            tensor.convolve(wavelet)


    def add_tag(self, tag):
       """ Appends string to tags list
       
       Tags can be used to support customized uses, such as storing metdata not
       included in mtuq.Station
       """
       if type(tag) not in [str, unicode]:
           raise TypeError

       for tensor in self:
           tensor.tags.append(tag)


    def remove_tag(self, tag):
       """ Removes string from tags list
       """
       for tensor in self:
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


    def as_array(self, components=['Z', 'R', 'T']):
        """ Returns time series from all stations in a single multidimensional
        array 

        Compared with iterating over streams and traces, provides a potentially
        faster way of accessing numeric trace data

        .. warning:

            Requires that all tensors have the same time discretization
            (or else an error is raised)

        """
        #check_time_sampling([stream[0] for stream in self])

        # array dimensions
        nc = len(components)
        nt = len(self[0][0])
        ns = len(self)
        nr = 0
        if self[0].include_mt:
            nr += 6
        if self[0].include_force:
            nr += 3

        array = np.zeros((nc,nr,ns,nt))
        for _i, tensor in enumerate(self):
            tensor.reset_components(components)
            # fill in array
            array[:, :, _i, :] = tensor._array
        return array




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
        Resets components returned by get_synthetics

        Suppose the vertical or radial components of the recorded data are
        found to be good but the transerve component is found to be bad.
        ``reset_components(['Z', 'R'])`` will cause ``get_synthetics`` to only
        return the first two components, avoiding unnecessary computation of
        the third
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
        Preallocates structures used by get_synthetics

        Every time get_synthetics is called, the numeric trace data gets
        overwritten. Every time reset_components is called, entire traces get
        overwritten.  The reference to the stream itself never changes.
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
        Precomputes numpy array used by get_synthetics
        """
        # the formulas relating the original time series to the linear
        # combination array vary depending on the scheme being used, so
        # are deferred to the subclass
        raise NotImplementedError("Must be implemented by subclass.")


    def get_synthetics(self, source):
        """
        Generates synthetics through a linear combination of Green's tensor
        times series
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


    def _allocate_cc(self, data, time_shift_max):
        """
        Allocates numpy arrays used by get_time_shift
        """
        nc = len(self.components)
        nr = 0

        if self.include_mt:
            nr += 6

        if self.include_force:
            nr += 3

        dt = self[0].stats.delta
        npts_padding = int(time_shift_max/dt)

        self._cc_sum = np.zeros(2*npts_padding+1)
        self._cc_all = np.zeros((nc, nr, 2*npts_padding+1))
        return self._cc_all


    def _compute_cc(self, data, time_shift_max):
        """
        Computes numpy arrays used by get_time_shift
        """
        if not hasattr(self, '_array'):
            self.reset_components(components=['Z','R','T'])

        array = self._array
        synthetics = self._synthetics

        cc = self._allocate_cc(data, time_shift_max)
        n1, n2, _ = cc.shape

        dt = self[0].stats.delta
        npts = self[0].stats.npts
        npts_padding = time_shift_max/dt

        for _i1, component in enumerate(self.components):
            if component in [trace.stats.channel[-1].upper() for trace in data]:
                trace = data.select(component=component)[0]
            else:
                continue

            for _i2 in range(n2):
                if (npts > 2000 or npts_padding > 200):
                    # for long traces or long lag times, frequency-domain
                    # implementation is usually faster
                    cc[_i1, _i2, :] =\
                        fftconvolve(trace.data, array[_i1, _i2, ::-1], 'valid')

                else:
                    # for short traces or short lag times, time-domain
                    # implementation is usually faster
                    cc[_i1, _i2, :] =\
                        np.correlate(trace.data, array[_i1, _i2, :], 'valid')


    def get_time_shift(self, data, source, group, time_shift_max):
        """ 
        Finds optimal time shift between the given data and synthetics
        generated from the given source

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
        try:
            cc_all = self._cc_all
            cc_sum = self._cc_sum
        except:
            self._compute_cc(data, time_shift_max)
            cc_all = self._cc_all
            cc_sum = self._cc_sum

        cc_sum[:] = 0.
        for component in group:
            _i = self.components.index(component)
            cc_sum += np.dot(source, cc_all[_i, :, :])

        npts_padding = (len(cc_sum)-1)/2
        return cc_sum.argmax() - npts_padding


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
        Convolves source wavelet with all time series

        :type wavelet: mtuq.util.wavelets.Wavelet
        :param wavelet: Source wavelet or source-time function to be used in
            convolution
        """
        for trace in self:
            wavelet.convolve(trace)


    def select(self, component=None, channel=None):
        """
        Return Stream with only those traces that match the supplied
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


    #
    # the remaining methods can be used to speed up trace data access in cases
    # where the time discretization is the same for all traces
    #
    def _check_time_sampling(self):
        """ Checks that time discretization is the same for all tensors
        """
        pass #check_time_sampling([stream[0] for stream in self])


    def as_array(self):
        """ Returns time series from all stations in a single multidimensional
        array 

        .. warning:

            This method requires that all tensors have the same time 
            discretization.

        .. note:

            Compared with iterating over obspy traces, this method provides a
            a potentially faster way of accessing numeric trace data.

        .. note:

            This method is used to supply input arrays for the C extension
            module `mtuq.grid_search._extensions`.

        """
        try:
            return self._array
        except:
            self._compute_array()
            return self._array


    def _allocate_array(self):
        """ Allocates numpy array that can be used for efficient synthetics
        generation
        """
        # array dimensions
        nc = 3
        nt = len(self[0][0])
        ns = len(self)
        nr = 0
        if self[0].include_mt:
            nr += 6
        if self[0].include_force:
            nr += 3

        self._array = np.zeros((nc, ns, nr, nt))

        return self._array


    def _compute_array(self):
        """ Computes numpy array that can be used for efficient synthetics
        generation
        """
        array = self._allocate_array()

        for _i, tensor in enumerate(self):
            # fill in 3D array of GreensTensor
            tensor.reset_components()

            # fill in 4D array of GreensTensorList
            array[:, _i, :, :] = tensor._array



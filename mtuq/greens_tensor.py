
import numpy as np

from copy import deepcopy
from mtuq.event import Origin
from mtuq.station import Station
from mtuq.dataset import Dataset
from mtuq.util.signal import check_time_sampling
from obspy.core import Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import fftconvolve


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
    def __init__(self, 
            traces=None, 
            station=None, 
            origin=None,
            id=None, 
            enable_force=False):

        for trace in traces:
            assert isinstance(trace, Trace)

        assert check_time_sampling(traces), NotImplementedError(
            "Time sampling differs from trace to trace.")

        super(GreensTensor, self).__init__(traces)

        if not isinstance(station, Station):
            raise TypeError

        if not isinstance(origin, Origin):
            raise TypeError

        if id:
            self.id = id
        else:
            self.id = station.id

        # station location and other metdata
        self.station = deepcopy(station)

        # event location and other metadata
        self.origin = origin

        self.distance_in_m, self.azimuth, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # enable force sources?
        self.enable_force = enable_force


    def initialize(self, components):
        """
        Computes arrays used by get_synthetics

        This method must be called prior to using get_synthetics or
        get_time_shift.

        Computes numpy arrays used in source-weighted linear combinations
        and allocates obspy stream used to hold the resulting synthetics. 

        This method can be rerun if the components change. Suppose a particular
        component of the recorded data is found to be corrupt. Then rerunning
        initialize without that component will cause get_synthetics to no longer
        return that component.
        """
        # Combining this method with __init__ would result in an endless 
        # recursion when the select method is called.  But more than that, the
        # work done by initialize is different and much more expensive than
        # __init__, so it makes sense to separate the two.

        raise NotImplementedError("Must be implemented by subclass")


    def allocate_synthetics(self):
        """
        Allocates obspy stream used to hold synthetics

        A single obspy stream is allocated by this method, and whenever
        get_synthetics is called, the numeric trace data gets overwritten.
        """
        self._synthetics = Stream()
        for channel in self.components:
            stats = deepcopy(self.station)
            stats.update({
                'npts': self[0].stats.npts,
                'channel': channel,
                })
            self._synthetics += Trace(np.zeros(stats.npts), stats)
        self._synthetics.id = self.id


    def get_synthetics(self, source):
        """
        Generates synthetics through a source-weighted linear combination
        """
        for _i, component in enumerate(self.components):
            s = self._synthetics[_i].data
            s[:] = 0.

            # we could use np.dot below, but speedup appears negligible
            if len(source)==6:
                # moment tensor source
                s += source[0]*self._array[_i, 0, :]
                s += source[1]*self._array[_i, 1, :]
                s += source[2]*self._array[_i, 2, :]
                s += source[3]*self._array[_i, 3, :]
                s += source[4]*self._array[_i, 4, :]
                s += source[5]*self._array[_i, 5, :]

            elif len(source)==3:
                # force source
                s += source[0]*self._array[_i, 6, :]
                s += source[1]*self._array[_i, 7, :]
                s += source[2]*self._array[_i, 8, :]

            else:
                raise TypeError

        return self._synthetics


    def initialize_cc(self, data, time_shift_max):
        """
        Computes cross correlations between data and Green's functions used by
        the get_time_shift method
        """
        if not hasattr(self, '_array'):
            raise Exception
        G = self._array
        n1,n2,npts = G.shape

        dt = self[0].stats.delta
        npts_padding = int(time_shift_max/dt)

        self._npts_padding = npts_padding
        self._array_cc_sum = np.zeros(2*npts_padding+1)
        self._array_cc_all = np.zeros((n1, n2, 2*npts_padding+1))

        # compute cross-correlations
        cc = self._array_cc_all
        for _i1, component in enumerate(self.components):
            d = data.select(component=component)[0].data

            for _i2 in range(n2):
                if (npts > 2000 or npts_padding > 200):
                    # for long traces or long lag times, frequency-domain
                    # implementation is usually faster
                    cc[_i1, _i2, :] = fftconvolve(d, G[_i1, _i2, ::-1], 'valid')

                else:
                    # for short traces or short lag times, time-domain
                    # implementation is usually faster
                    cc[_i1, _i2, :] = np.correlate(d, G[_i1, _i2, :], 'valid')


    def get_time_shift(self, data, source, group, time_shift_max):
        """ 
        Finds optimal time-shift correction between synthetics and
        user-supplied da

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
        if not hasattr(self, '_array_cc_all'):
            self.initialize_cc(data, time_shift_max)

        npts_padding = self._npts_padding
        cc_all = self._array_cc_all
        cc_sum = self._array_cc_sum
        cc_sum[:] = 0.

        for component in group:
            _i = self.components.index(component)
            if len(source)==6:
                cc_sum += source[0] * cc_all[_i, 0, :]
                cc_sum += source[1] * cc_all[_i, 1, :]
                cc_sum += source[2] * cc_all[_i, 2, :]
                cc_sum += source[3] * cc_all[_i, 3, :]
                cc_sum += source[4] * cc_all[_i, 4, :]
                cc_sum += source[5] * cc_all[_i, 5, :]
            elif len(source)==3:
                # force source
                cc_sum += source[0] * cc_all[_i, 6, :]
                cc_sum += source[1] * cc_all[_i, 7, :]
                cc_sum += source[2] * cc_all[_i, 8, :]


        # what is the index of the maximum element of the padded array?
        argmax = cc_sum.argmax()

        # what is the associated cross correlation lag?
        ioff = argmax-npts_padding

        return ioff


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
            enable_force=self.enable_force)


    def convolve(self, wavelet):
        """
        Convolves source wavelet with all time series

        :type wavelet: mtuq.util.wavelets.Wavelet
        :param wavelet: Source wavelet or source-time function to be used in
            convolution
        """
        for trace in self:
            wavelet.convolve(trace)


    def select(self, *args, **kwargs):
        """
        Return new GreensTensor with only those traces that match the supplied
        metadata criteria
        """
        # Inherited method "select" doesn't work because it tries to return a
        # a Stream rather than GreensTensor. To fix it, we need to manually 
        # convert to Stream, call select, then convert back to a GreensTensor
        stream = Stream([trace for trace in self]).select(*args, **kwargs)
        return self.__class__(
            [trace for trace in stream],
            station=self.station,
            origin=self.origin,
            id=self.id,
            enable_force=self.enable_force)


    def __add__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")


    def __iadd__(self, *args):
        raise Exception("It doesn't make sense to add time series to "
           " a GreensTensor")



class GreensTensorList(list):
    """ Container for one or more GreensTensor objects

    Basically, a list of GreensTensors. Very similar to an MTUQ Dataset, except 
    rather than a list of Streams containing  observed data, holds synthetic 
    Green's tensors
    """
    def __init__(self, greens_tensors=[], id=None):
        # typically the id is the event name, event origin time, or some other
        # attribute shared by all GreensTensors
        self.id = id

        for greens_tensor in greens_tensors:
            self += [greens_tensor]


    def get_synthetics(self, mt):
        """ Generates synthetic seismograms by summing Green's functions in a 
        linear combination weighted by moment tensor elements

        :param mt: Moment tensor to be used in linear combination
        :rtype: An MTUQ ``Dataset`` in which all streams correspond to the
            given moment tensor, and each each individual stream corresponds to
            an individual station
        """
        synthetics = Dataset()
        for greens_tensor in self:
            synthetics += [greens_tensor.get_synthetics(mt)]
        return synthetics


    def as_array(self):
        """ Returns time series from all Green's tensors in a single 
        multidimensional array
        """
        raise NotImplementedError

        #if self._ndarray:
        #    return self._ndarray

        #for greens_tensor in self:
        #    pass

        #self._ndarray = ndarray
        #return ndarray


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
        for greens_tensor in self:
            processed +=\
                [greens_tensor.apply(function, *args, **kwargs)]
        return processed


    def map(self, function, *sequences):
        """ Applies a function in-pace to each GreensTensor in the list. If one
        or more optional sequences are given, the function is called with an 
        argument list consisting of the corresponding item of each sequence. 
        Similar to the behavior of the python built-in "map".
        """
        processed = GreensTensorList()
        for _i, greens_tensor in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed +=\
                [greens_tensor.apply(function, *args)]
        return processed


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        convolved = GreensTensorList()
        for greens_tensor in self:
            convolved += [greens_tensor.convolve(wavelet)]
        return convolved


    def add_tag(self, tag):
       """ Appends string to tags list
       
       Tags can be used to support customized uses, such as storing metdata not
       included in mtuq.Station
       """
       if type(tag) not in [str, unicode]:
           raise TypeError

       for greens_tensor in self:
           greens_tensor.tags.append(tag)


    def remove_tag(self, tag):
       """ Removes string from tags list
       """
       for greens_tensor in self:
           greens_tensor.tags.remove(tag)


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





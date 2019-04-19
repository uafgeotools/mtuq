
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
    """
    Green's tensor base class

    Holds multiple time series corresponding to the independent elements 
    of an elastic Green's tensor. 

    .. note::

        Besides the methods described below, GreensTensor also includes
        data processing methods inherited from ``obspy.Stream``.
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


    def initialize(self, components=['Z', 'R', 'T']):
        """
        Prepares structures used by get_synthetics

        Allocates and computes numpy array used by get_synthetics and
        allocates obspy stream to hold the resulting synthetic data

        This method can be rerun if the components change. For example if a
        particular component of the recorded data is found to be corrupt, 
        rerunning initialize without that component will cause get_synthetics
        to no longer return that component.
        """

        # The idea is for __init__ to do only the inexpensive work of 
        # instantiating a GreensTensor, and defer all the expensive work to 
        # this method

        # Another reason to separate the two methods is that combining them
        # would cause an endless recursion when select is called. This problem
        # is not unique to mtuq, but common to any object that has
        # ``obspy.core.Stream`` as a parent

        # sets components attribute
        self._set_components(components)

        # allocates obspy stream used by get_synthetics
        self._allocate_synthetics()

        # allocates and computes numpy array used by get_synthetics
        self._allocate_array()
        self._compute_array()


    def _set_components(self, components):
        """
        Checks input argument and sets component attribute
        """
        if components==None:
            components = []
        for component in components:
            assert component in ['Z', 'R', 'T']
        self.components = components


    def _allocate_synthetics(self):
        """
        Allocates obspy stream used by get_synthetics
        """
        npts = self[0].stats.npts

        # a single obspy stream is allocated, then every time get_synthetics
        # is called, the stream gets overwritten
        self._synthetics = Stream()
        self._synthetics.id = self.id

        for component in self.components:
            # add stats object
            stats = deepcopy(self.station)
            stats.update({'npts': npts, 'channel': component})

             # add trace object
            self._synthetics += Trace(np.zeros(stats.npts), stats)


    def _allocate_array(self):
        """
        Allocates numpy array used by get_synthetics
        """
        nt = self[0].stats.npts
        nc = len(self.components)
        nr = 6
        if self.enable_force:
            nr += 3
        self._array = np.zeros((nc, nr, nt))


    def _compute_array(self):
        """
        Computes numpy array used by get_synthetics
        """
        # the formulas relating the original time series to the linear
        # combination array vary depending on the scheme being used, so
        # are deferred to the subclass
        raise NotImplementedError("Must be implemented by subclass.")


    def get_synthetics(self, source):
        """
        Generates synthetics through a linear combination of Green's tensor
        times series weighted by source elements
        """
        try:
            array = self._array
            synthetics = self._synthetics
        except:
            raise Exception(
               "initialize() must be called prior to generating synthetics")

        for _i, component in enumerate(self.components):
            s = synthetics[_i].data
            s[:] = 0.

            # we could use np.dot below, but it actually appears slower
            if len(source)==6:
                # moment tensor source
                s += source[0]*array[_i, 0, :]
                s += source[1]*array[_i, 1, :]
                s += source[2]*array[_i, 2, :]
                s += source[3]*array[_i, 3, :]
                s += source[4]*array[_i, 4, :]
                s += source[5]*array[_i, 5, :]

            elif len(source)==3:
                # force source
                s += source[0]*array[_i, 6, :]
                s += source[1]*array[_i, 7, :]
                s += source[2]*array[_i, 8, :]

            else:
                raise TypeError

        return synthetics


    def _allocate_cc(self, data, time_shift_max):
        """
        Allocates numpy arrays used by get_time_shift
        """
        nc = len(self.components)
        nr = 6
        if self.enable_force:
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
        try:
            array = self._array
        except:
            raise Exception(
                "initialize() must be called prior to computing time shifts")

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
        Finds optimal time-shift between the given data and synthetics
        corresponding to the given source

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
            if len(source)==6:
                cc_sum += source[0] * cc_all[_i, 0, :]
                cc_sum += source[1] * cc_all[_i, 1, :]
                cc_sum += source[2] * cc_all[_i, 2, :]
                cc_sum += source[3] * cc_all[_i, 3, :]
                cc_sum += source[4] * cc_all[_i, 4, :]
                cc_sum += source[5] * cc_all[_i, 5, :]
            elif len(source)==3:
                cc_sum += source[0] * cc_all[_i, 6, :]
                cc_sum += source[1] * cc_all[_i, 7, :]
                cc_sum += source[2] * cc_all[_i, 8, :]

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
        # a Stream rather than a GreensTensor. To fix it, we need to manually 
        # convert to a Stream, call select, then convert back
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
    """

    def __init__(self, greens_tensors=[], id=None):
        # typically the id is the event name, event origin time, or some other
        # attribute shared by all GreensTensors
        self.id = id

        for greens_tensor in greens_tensors:
            self += [greens_tensor]


    def get_synthetics(self, mt):
        """ Generates synthetic seismograms by summing Green's functions 
        weighted by moment tensor elements

        :param mt: Moment tensor to be used in linear combination
        """
        synthetics = Dataset()
        for greens_tensor in self:
            synthetics += [greens_tensor.get_synthetics(mt)]
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
        for greens_tensor in self:
            processed +=\
                [greens_tensor.apply(function, *args, **kwargs)]
        return self.__class__(processed)


    def map(self, function, *sequences):
        """ Applies a function in-pace to each GreensTensor in the list. If one
        or more optional sequences are given, the function is called with an 
        argument list consisting of the corresponding item of each sequence. 
        Similar to the behavior of the python built-in "map".
        """
        processed = []
        for _i, greens_tensor in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed +=\
                [greens_tensor.apply(function, *args)]
        return self.__class__(processed)


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        convolved = []
        for greens_tensor in self:
            convolved += [greens_tensor.convolve(wavelet)]
        return self.__class__(convolved)



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



class maGreensTensorList(GreensTensorList):
    """ Specialized GreensTensorList subclass

    Adds multidimensional array machinery that can be used for implementing 
    functions that act on numpy arrays rather than obspy streams.

    .. warning:

        Unlike its parent class, this subclass requires all tensors have the 
        same time discretization.
    """

    def __init__(self, greens_tensors=[], id=None, mask=None):
        super(maGreensTensorList, self).__init__(greens_tensors, id)

        # this attribute is not yet impelemented
        self.mask = mask

        self._check_time_sampling()


    def _check_time_sampling(self):
        """ Checks that time discretization is the same for all tensors
        """
        pass
        #check_time_sampling([stream[0] for stream in self])


    def _allocate_array(self):
        """ Allocates numpy array that can be used for efficient synthetics
        generation
        """
        # array dimensions
        nc = 3
        nt = len(self[0][0])
        ns = len(self)
        nr = 6

        if self[0].enable_force:
            nr += 3

        self._array = np.zeros((nc, ns, nr, nt))


    def _compute_array(self):
        """ Computes numpy array that can be used for efficient synthetics
        generation
        """
        for _i, greens_tensor in enumerate(self):
            # fill in 3D array of GreensTensor
            greens_tensor.initialize()

            # fill in 4D array of GreensTensorList
            self._array[:, _i, :, :] = greens_tensor._array


    def _allocate_cc(self, npts):
        """ Allocates numpy array that can be used for efficient time shift
        calculations
        """
        self._npts_cc = npts

        # array dimensions
        nc = 3
        nt = len(self[0][0])
        ns = len(self)
        nr = 6

        if self[0].enable_force:
            nr += 3

        # allocate arrays
        self._cc_all = np.zeros((nc, ns, nr, 2*npts+1))


    def _compute_cc(self, data, time_shift_max):
        """ Computes numpy array that can be used for efficient time shift
        calculations
        """
        # allocate numpy array
        dt = self[0][0].stats.delta
        self._allocate_cc(time_shift_max/dt)

        for _i, greens_tensor in enumerate(self):
            # fills 3D array of GreensTensor
            greens_tensor._compute_cc(data, time_shift_max)

            # fills 4D array of GreensTensorList
            self._cc_all[_i, :, :, :] = greens_tensor._cc_all



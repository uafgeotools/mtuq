
import obspy
import numpy as np

from copy import copy
from obspy.geodetics import gps2dist_azimuth



class Dataset(list):
    """ Seismic data container

    A list of ObsPy streams in which each stream corresponds to a single
    seismic station

    .. note::

        Each supported file format has a corresponding reader that creates an
        Dataset (see ``mtuq.io.readers``).

    """
    def __init__(self, streams=[], stations=[], origins=[],
                 id=None, tags=[]):
        """ Constructor
        """
        if len(stations)!=len(streams):
            raise Exception

        if len(origins)!=len(streams):
            raise Exception

        for _i, stream in enumerate(streams):

            # collect location information
            (stream.preliminary_distance_in_m,
            stream.preliminary_azimuth, _) =\
                gps2dist_azimuth(
                    origins[_i].latitude,
                    origins[_i].longitude,
                    stations[_i].latitude,
                    stations[_i].longitude)

            # create unique identifier
            stream.id = '.'.join([
                stations[_i].network,
                stations[_i].station,
                stations[_i].location])

            # append tags to stream
            if not hasattr(stream, 'tags'):
                stream.tags = list()
            stream.tags.extend(copy(tags))

            # append stream to list
            self.append(stream)

        self.stations = stations
        self.origins = origins
        self.id = id


    def apply(self, function, *args, **kwargs):
        """ Applies function to all streams in the dataset

        Applies a function to each Stream, similar to the Python built-in
        ``apply``.
        """
        processed = []
        for stream in self:
            processed += [function(stream, *args, **kwargs)]

        return self.__class__(
            processed, self.stations, self.origins, id=self.id)


    def map(self, function, *sequences):
        """ Maps function to all streams in the dataset

        Applies a function to each Stream. If one or more optional sequences 
        are given, the function is called with an argument list consisting of
        corresponding items of each sequence. Similar to the Python built-in
        ``map``.
        """
        processed = []
        for _i, stream in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed += [function(stream, *args)]

        return self.__class__(
            processed, self.stations, self.origins, id=self.id)


    def max(self):
        """ Returns maximum absolute amplitude over all traces
        """
        max_all = -np.inf
        for stream in self:
            for trace in stream:
                if not getattr(trace, 'weight', 1.):
                    continue
                if trace.data.max() > max_all:
                    max_all = abs(trace.data).max()
        return max_all


    def sort_by_distance(self, reverse=False):
        """ Sorts in-place by hypocentral distance
        """
        self.sort_by_function(lambda data: data.preliminary_distance_in_m,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """ Sorts in-place by source-receiver azimuth
        """
        self.sort_by_function(lambda data: data.preliminary_azimuth,
            reverse=reverse)


    def sort_by_function(self, function, reverse=False):
        """ Sorts in-place by user-supplied function
        """
        wrapped_function = lambda x: function(x[0])
        _, self.stations, self.origins = zip(*sorted(zip(
            [stream for stream in self], self.stations, self.origins),
            key=wrapped_function, reverse=reverse))

        self.sort(key=function, reverse=reverse)


    def get_stations(self):
        """ Extracts station metadata from all streams in list
        """
        return self.stations


    def get_origins(self):
        """ Returns preliminary origin location and time
        """
        return self.origins


    def add_tag(self, tag):
       """ Appends string to tags list
       
       Tags can be used to support customized uses, such as storing metdata not
       included in mtuq.Station
       """
       if type(tag) not in [str, unicode]:
           raise TypeError

       for stream in self:
           stream.tags.append(tag)


    def remove_tag(self, tag):
       """ Removes string from tags list
       """
       for stream in self:
           stream.tags.remove(tag)


class maDataset(Dataset):
    """ Specialized Dataset subclass

    Adds multidimensional array machinery that can be used for implementing 
    functions that act on numpy arrays rather than obspy streams.

    .. warning:

        Unlike the parent class, this subclass requires all streams have the 
        same time discretization.

    """
    def __init__(self, *args, **kwargs):
        super(maDataset, self).__init__(*args, **kwargs)

        # this method is not yet implemented
        self._check_time_sampling()


    def _check_time_sampling(self):
        """ Checks that time discretization is the same for all stations
        """
        pass


    def get_array(self):
        """ Returns time series from all stations and components in a single 
        multidimensional array
        """
        try:
            return self._array
        except:
            self._allocate_array()
            return self._array


    def get_array_mask(self):
        raise NotImplementedError


    def _allocate_array(self):
        # count number of nonempty streams
        ns = 0
        for stream in self:
            if len(stream)==0:
                ns += 1
        nt = self[0][0].stats.npts

        # allocate array
        self._array = np.zeros((3, ns, nt))
        array = self._array

        _i = 0
        for stream in self:
            if len(stream)==0:
                continue
            try:
                trace = stream.select(component='Z')
                array[0, _i, :] = trace.data
            except:
                pass
            try:
                trace = stream.select(component='R')
                array[1, _i, :] = trace.data
            except:
                pass
            try:
                trace = stream.select(component='T')
                array[2, _i, :] = trace.data
            except:
                pass
            _i += 1



def EventDataset(streams=[], stations=[], origin=None,            
                 id=None, tags=[]):
    """ Returns single-event seismic data container

    Returns a Dataset object in which each stream corresponds to a single
    station and all streams correspond to the same event

    """
    size = len(streams)

    if len(stations)!=size:
        raise Exception

    if not origin:
        raise Exception

    origins = []
    for _ in range(size):
        origins += [copy(origin)]

    return maDataset(streams, stations, origins, id, tags)



import obspy
import numpy as np
from copy import copy


class Dataset(list):
    """ Seismic data container

    A list of ObsPy streams in which each stream corresponds to a single
    seismic station and all streams correspond to a single event.

    .. note::

        Each supported file format has a corresponding reader that creates an
        MTUQ Dataset (see ``mtuq.io.readers``).

    """
    def __init__(self, streams=[], stations=[], origins=[],
                 id=None, tags=[]):
        """ Constructor
        """
        size = len(streams)

        if len(stations)!=size:
            raise Exception

        if len(origins)!=size:
            raise Exception

        for _i, stream in enumerate(streams):
            stream.stats = stations[_i]
            stream.id = stations[_i].id

            if not hasattr(stream, 'tags'):
                stream.tags = []
            for tag in tags:
                stream.tags.append(tag)

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
        """ Returns the maximum absolute amplitude of all traces
        """
        max_all = -np.inf
        for stream in self:
            for trace in stream:
                weight = getattr(trace, 'weight', 1.)
                if not weight:
                    continue
                if trace.data.max() > max_all:
                    max_all = trace.data.max()
        return max_all


    def sort_by_distance(self, reverse=False):
        """ Sorts in-place by hypocentral distance
        """
        self.sort_by_function(lambda data: data.stats.preliminary_distance_in_m,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """ Sorts in-place by source-receiver azimuth
        """
        self.sort_by_function(lambda data: data.stats.preliminary_azimuth,
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

    return Dataset(streams, stations, origins, id, tags)



import obspy
import numpy as np
from copy import copy


class Dataset(object):
    """ Seismic data container

    A list of ObsPy streams in which each stream corresponds to a single
    seismic station and all streams correspond to a single event.

    .. note::

        Each supported file format has a corresponding reader that creates an
        MTUQ Dataset (see ``mtuq.io.readers``).

    """
    def __init__(self, streams=[], stations=[], preliminary_origin=None,
                 station_ids=[], event_id=None, tags=[]):
        """ Constructor
        """
        self.__list__ = []
        for _i, stream in enumerate(streams):
            self.__list__ += [stream]
            self.__list__[_i].tags = copy(tags)
            
            # add station-related attributes
            self.__list__[_i].id = station_ids[_i]
            self.__list__[_i].stats = stations[_i]

        # add event-related attributes
        self.origin = preliminary_origin
        self.id = event_id



    # the next two methods can be used to apply signal processing operations or
    # other functions to the dataset
    def apply(self, function, *args, **kwargs):
        """ Applies function to all streams in the dataset

        Applies a function to each Stream, similar to the Python built-in
        ``apply``.
        """
        processed =\
            self.__class__(event_id=self.id, preliminary_origin=self.origin)
        for stream in self:
            processed += function(stream, *args, **kwargs)
        return processed


    def map(self, function, *sequences):
        """ Maps function to all streams in the dataset

        Applies a function to each Stream. If one or more optional sequences 
        are given, the function is called with an argument list consisting of
        corresponding items of each sequence. Similar to the Python built-in
        ``map``.
        """
        processed =\
            self.__class__(event_id=self.id, preliminary_origin=self.origin)
        for _i, stream in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed += function(stream, *args)
        return processed


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



    # various sorting methods
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
        self.__list__.sort(key=function, reverse=reverse)


    # metadata extraction methods
    def get_stations(self):
        """ Extracts station metadata from all streams in list
        """
        stations  = []
        for stream in self:
            stations += [stream.stats]
        return stations


    def get_origin(self):
        """ Returns preliminary origin location and time
        """
        return self.origin


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



    # the remaining methods deal with indexing and iteration over the dataset
    def __add__(self, item):
        self.__list__ += [item]
        return self


    def _get_index(self, id):
        for index, stream in enumerate(self.__list__):
            if id==stream.id:
                return index

    def __iter__(self):
        return self.__list__.__iter__()


    def __getitem__(self, index):
        return self.__list__[index]


    def __setitem__(self, index, value):
        self.__list__[index] = value


    def __len__(self):
        return len(self.__list__)



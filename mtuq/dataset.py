
import obspy
import numpy as np
import warnings

from copy import copy
from mtuq.event import Origin
from mtuq.station import Station
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth



class Dataset(list):
    """ Seismic data container

    A list of ObsPy streams in which each stream corresponds to a single
    seismic station

    .. note::

        Each supported file format has a corresponding reader that creates a 
        Dataset (see ``mtuq.io.readers``).

    """
    def __init__(self, streams=[], id=None, tags=[]):
        """ Constructor method
        """
        self.id = id

        for stream in streams:
            self.append(stream)

        for tag in copy(tags):
            self.tag_add(tag)



    def append(self, stream):
        """ Appends stream to Dataset
        """
        assert issubclass(type(stream), Stream),\
            ValueError("Only Streams can be appended to a Dataset")

        # create unique identifier
        try:
            stream.id = '.'.join([
                stream.station.network,
                stream.station.station,
                stream.station.location])
        except:
            stream.id = '.'.join([
                stream[0].stats.network,
                stream[0].stats.station,
                stream[0].stats.location])

        if not hasattr(stream, 'tags'):
            stream.tags = list()

        if not hasattr(stream, 'station'):
            warnings.warn("Stream lacks station metadata")
        elif not hasattr(stream, 'origin'):
            warnings.warn("Stream lacks origin metadata")
        else:
            (stream.distance_in_m, stream.azimuth, _) =\
                gps2dist_azimuth(
                    stream.origin.latitude,
                    stream.origin.longitude,
                    stream.station.latitude,
                    stream.station.longitude)

        super(Dataset, self).append(stream)


    def select(self, selector):
        """ Selects streams that match the given station or origin
        """
        if type(selector) is Station:
           selected = lambda stream: stream.station==selector

        elif type(selector) is Origin:
           selected = lambda stream: stream.origin==selector

        elif type(selector) is list:
           selected = lambda stream: stream.id in selector

        else:
            raise ValueError(
                "SELECTOR must be a Station, Origin or list")

        return self.__class__(
            id=self.id, streams=filter(selected, self))


    def apply(self, function, *args, **kwargs):
        """ Applies a function to all streams

        .. warning ::

            Although ``apply`` returns a new Dataset, contents of the original 
            Dataset may still be overwritten when applying certain functions.

            If you are unsure of the behavior of the function and wish to 
            preserve the original Dataset, make a `deepcopy` of it first.

            (Deep copies are not necessary when using `mtuq.process_data`, 
            because the original trace data are preserved by default.
            This behavior can be overridden by manually supplying 
            `inplace=True` as a keyword argument.)

        """
        processed = []
        for stream in self:
            processed += [function(stream, *args, **kwargs)]

        return self.__class__(
            processed, id=self.id)


    def map(self, function, *sequences):
        """ Maps function to all streams

        Maps a function to all streams in the Dataset. If one or more optional
        sequences are given, the function is called with an argument list 
        consisting of corresponding items of each sequence, just like the 
        Python built-in ``map``.

        .. warning ::

            Although ``map`` returns a new Dataset, contents of the original 
            Dataset may still be overwritten when mapping certain functions.

            If you are unsure of the behavior of the function and wish to 
            preserve the original Dataset, make a `deepcopy` of it first. 

            (Deep copies are not necessary when using `mtuq.process_data`, 
            because the original trace data are preserved by default.
            This behavior can be overridden by manually supplying 
            `inplace=True` as a keyword argument.)

        """
        processed = []
        for _i, stream in enumerate(self):
            args = [sequence[_i] for sequence in sequences]
            processed += [function(stream, *args)]

        return self.__class__(
            processed, id=self.id)


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
        self.sort_by_function(lambda data: data.distance_in_m,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """ Sorts in-place by source-receiver azimuth
        """
        self.sort_by_function(lambda data: data.azimuth,
            reverse=reverse)


    def sort_by_function(self, function, reverse=False):
        """ Sorts in-place by user-supplied function
        """
        self.sort(key=function, reverse=reverse)


    def get_stations(self):
        """ Returns station metadata from all streams as a `list` of
        `mtuq.station.Stations` objects

        .. note ::

          For Datasets created using ``mtuq.io.readers``, SAC header metadata
          is used to populate the Station attributes

        """
        stations = []
        for stream in self:
            stations += [stream.station]
        return stations


    def get_origins(self):
        """ Returns origin metadata from all streams as a `list` of 
        `mtuq.event.Origin` objects

        .. note ::

          - For Datasets created using ``mtuq.io.readers.sac``, origin 
            metadata represent catalog information read from SAC headers

          - For Datasets created using ``GreensTensor.get_synthetics``,
            origin metadata are inherited from the GreensTensor

        """
        origins = []
        for stream in self:
            origins += [stream.origin]

            if getattr(self, '_warnings', True):
                if stream.origin!=self[0].origin:
                    warnings.warn(
                        "Different streams in the Dataset correpond to "
                        "different events.\n\n"
                        "This may be intentional. Feel free to disable this "
                        "warning by setting Dataset._warnings=False")

        return origins


    def tag_add(self, tag):
       """ Appends string to tags list
       
       Tags can be used to support customized uses, such as storing metdata not
       included in ``Station`` or ``Origin`` objects
       """
       if type(tag)!=str:
           raise TypeError

       for stream in self:
           if tag not in stream.tags:
               stream.tags.append(tag)


    def tag_remove(self, tag):
       """ Removes string from tags list
       """
       for stream in self:
           if tag in stream.tags:
               stream.tags.remove(tag)



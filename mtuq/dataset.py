
import obspy
import os
import numpy as np
import pickle

from copy import copy, deepcopy
from mtuq.event import Origin
from mtuq.station import Station
from mtuq.util import warn
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth



class Dataset(list):
    """ Seismic data container

    A list of ObsPy streams in which each stream corresponds to a single
    seismic station

    .. note::

        Each supported file format has a corresponding reader that creates
        Datasets (see ``mtuq.io.readers``).

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
        if hasattr(stream, 'station'):
            stream.id = '.'.join([
                stream.station.network,
                stream.station.station,
                stream.station.location])
        elif len(stream) > 0:
            stream.id = '.'.join([
                stream[0].stats.network,
                stream[0].stats.station,
                stream[0].stats.location])
        else:
            stream.id = ''

        if not hasattr(stream, 'tags'):
            stream.tags = list()

        # optional origin and station metadata warnings
        _warnings = getattr(self, '_warnings', False)

        if not hasattr(stream, 'station') and _warnings:
            warn("Stream lacks station metadata")

        if not hasattr(stream, 'origin') and _warnings:
            warn("Stream lacks origin metadata")

        if hasattr(stream, 'station') and hasattr(stream, 'origin'):
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
                "`selector` must be a `Station`, `Origin` or list thereof")

        return self.__class__(
            id=self.id, streams=filter(selected, self))


    def apply(self, function, *args, **kwargs):
        """ Applies a function to all streams

        .. warning ::

            Although ``apply`` returns a new `Dataset`, contents of the
            original `Dataset` may still be overwritten, depending on
            the function. To preserve the original, consider making a 
            `copy` first.


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

            Although ``map`` returns a new `Dataset`, contents of the
            original `Dataset` may still be overwritten, depending on
            the function. To preserve the original, consider making a 
            `copy` first.

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


    def get_components(self):
        """ Returns `list` of components from all streams
        """
        components = []
        for stream in self:
            components += [[]]
            for trace in stream:
                components[-1] += [trace.stats.channel[-1].upper()]
        return components


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
                    warn(
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


    def __copy__(self):
        try:
            new_id = self.id+'_copy'
        except:
            new_id = None

        new_ds = type(self)(id=new_id)
        for stream in self:
            new_ds.append(deepcopy(stream))
        return new_ds


    def write(self, path, format='sac'):
        """ Writes a Python pickle of current dataset
        """
        if format.lower() == 'pickle':

            with open(filename, "wb") as file:
               pickle.dump(self, path)

        elif format.lower() == 'sac':

            os.makedirs(path, exist_ok=True)
            for stream in self:
                for trace in stream:

                    keys = ('network','station','location','channel')
                    filename = '.'.join([trace.stats[key] for key in keys])

                    fullpath = '%s/%s.%s' % (path,filename,'sac')
                    trace.write(fullpath, format='sac')
                    
        


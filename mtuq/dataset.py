
import obspy
import numpy as np
import warnings

from copy import copy
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth



class Dataset(list):
    """ Seismic data container

    A list of ObsPy streams in which each stream corresponds to a single
    seismic station

    .. note::

        Each supported file format has a corresponding reader that creates an
        Dataset (see ``mtuq.io.readers``).

    """
    def __init__(self, streams=[], id=None, tags=[]):
        """ Constructor method
        """
        self.id = id

        for _i, stream in enumerate(streams):
            self.append(stream)

        for tag in copy(tags):
            self.add_tag(tag)



    def append(self, stream):
        """ Appends stream to dataset
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


    def apply(self, function, *args, **kwargs):
        """ Applies function to all streams in the dataset

        Applies a function to each stream, similar to the Python built-in
        ``apply``.
        """
        processed = []
        for stream in self:
            processed += [function(stream, *args, **kwargs)]

        return self.__class__(
            processed, id=self.id)


    def map(self, function, *sequences):
        """ Maps function to all streams in the dataset

        Applies a function to each stream. If one or more optional sequences 
        are given, the function is called with an argument list consisting of
        corresponding items of each sequence. Similar to the Python built-in
        ``map``.
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
        """ Returns station attribute of each stream

        For Datasets created using ``mtuq.io.readers``, SAC headers or
        other file metadata are used to populate the Station attributes
        """
        stations = []
        for stream in self:
            stations += [stream.station]
        return stations


    def get_origins(self):
        """ Returns origin attribute of each stream

        For Datasets created using ``mtuq.io.readers``, preliminary event
        metadata (e.g. catalog information) is used to define the Origins.
        For Datasets created using ``get_synthetics`` methods, solver/
        IO client input is used to define the Origins
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


    def add_tag(self, tag):
       """ Appends string to tags
       
       Tags can be used to support customized uses, such as storing metdata not
       included in ``Station`` or ``Origin`` objects
       """
       if type(tag) not in [str, unicode]:
           raise TypeError

       for stream in self:
           if tag not in stream.tags:
               stream.tags.append(tag)


    def remove_tag(self, tag):
       """ Removes string from tags list
       """
       for stream in self:
           if tag in stream.tags:
               stream.tags.remove(tag)


    #
    # the remaining methods can be used to speed up trace data access in cases
    # where the time discretization is the same for all traces
    #
    def _check_time_sampling(self):
        """ Checks if time discretization is the same for all traces
        """
        pass


    def as_array(self, components=['Z','R','T']):
        """ Returns all numeric trace data in a single NumPy array

        Compared with iterating over streams and traces, provides a potentially
        faster way of accessing numeric trace data

        .. warning::

            Requires that all tensors have the same time discretization
            (or else an error is raised)

        """
        nc = len(components)

        # count number of nonempty streams
        ns = 0
        for stream in self:
            if len(stream)==0:
                ns += 1
        nt = self[0][0].stats.npts

        # allocate array
        array = np.zeros((3, ns, nt))

        _i = 0
        for _i, stream in enumerate(self):
            if 'Z' in components:
                try:
                    trace = stream.select(component='Z')[0]
                    array[0, _i, :] = trace.data
                except:
                    pass
            if 'R' in components:
                try:
                    trace = stream.select(component='R')[0]
                    array[1, _i, :] = trace.data
                except:
                    pass
            if 'T' in components:
                try:
                    trace = stream.select(component='T')[0]
                    array[2, _i, :] = trace.data
                except:
                    pass


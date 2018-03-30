
import obspy

class DatasetBase(object):
    """ Seismic data container

        Basically, a list of obspy streams. Each stream corresponds to a
        single seismic station and holds all the components recorded at that
        station.  Provides methods that help with data processing and metadata
        extraction.

        Each supported file format will have a corresponding reader utility
        that creates an MTUQ Dataset from files stored in that format.  For an
        example, see mtuq.dataset.sac.reader
    """

    def __init__(self, streams=None, id=None):
        # typically the id is the event name, event origin time, or some other
        # attribute shared by all streams
        self.id = id

        self.__list__ = []

        if not streams:
            # return an empty container; streams can be added later on
            return

        for stream in streams:
            self.__add__(stream)


    # the next two methods can be used to apply signal processing operations or
    # other functions to the dataset
    def apply(self, function, *args, **kwargs):
        """
        Returns the result of applying a function to each Stream in the 
        dataset. Similar to the behavior of the python built-in "apply".
        """
        processed = self.__class__(id=self.id)
        for stream in self.__list__:
            processed += function(stream, *args, **kwargs)
        return processed


    def map(self, function, *sequences):
        """
        Returns the result of applying a function to each Stream in the
        dataset. If one or more optional sequences are given, the function is 
        called with an argument list consisting of the corresponding item of
        each sequence. Similar to the behavior of the python built-in "map".
        """
        processed = self.__class__(id=self.id)
        for _i, stream in enumerate(self.__list__):
            args = [sequence[_i] for sequence in sequences]
            processed += function(stream, *args)
        return processed


    # various sorting methods
    def sort_by_distance(self, reverse=False):
        """ 
        Sorts the dataset in-place by hypocentral distance
        """
        self.sort_by_function(lambda data: data.station.catalog_distance,
            reverse=reverse)


    def sort_by_azimuth(self, reverse=False):
        """
        Sorts the dataset in-place by hypocentral azimuth
        """
        self.sort_by_function(lambda data: data.station.catalog_azimuth,
            reverse=reverse)


    def sort_by_function(self, function, reverse=False):
        """ 
        Sorts the dataset in-place using the python built-in "sort"
        """
        self.__list__.sort(key=function, reverse=reverse)


    # because the way metadata are organized in obspy streams depends on file
    # format, the next two methods are deferred to the subclass
    def get_origin(self):
        """
        Extracts origin information from metadata
        """
        raise NotImplementedError("Must be implemented by subclass")


    def get_station(self):
        """
        Extracts station information from metadata
        """
        raise NotImplementedError("Must be implemented by subclass")


    # the next method is called repeatedly during class creation
    def __add__(self, stream):
        assert hasattr(stream, 'id')
        assert isinstance(stream, obspy.Stream)
        self.__list__.append(stream)
        try:
            stream.station = self.get_station()
            stream.catalog_origin = self.get_origin()
        except:
            pass
        return self


    def remove(self, id):
        index = self._get_index(id)
        self.__list__.pop(index)


    # the remaining methods deal with indexing and iteration over the dataset
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




import obspy

class DatasetBase(object):
    """ Seismic data container

        Basically, a list of obspy streams. Each stream corresponds to a
        single seismic station and holds all the components recorded at that
        station.  Provides methods that help with data processing and metadata
        extraction.

        The work of generating a Dataset is carried out by a "reader";
        for example, see mtuq.dataset.sac.reader
    """

    def __init__(self, data=None, id=None):
        self.__list__ = []

        # event name or other unique identifier
        self.id = id

        if not data:
            return

        for stream in data:
            self.__add__(stream)


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


    def sort(self, *args, **kwargs):
        self.__list__.sort(*args, **kwargs)


    def sort_by_distance(self, stations, reverse=False):
        raise NotImplementedError


    def sort_by_azimuth(self, stations, reverse=False):
        raise NotImplementedError


    # because the way metadata are organized in obspy streams depends on file
    # format, the next two methods are deferred to the subclass
    def get_origin(self):
        """
        Extracts origin information from metadata
        """
        raise NotImplementedError("Must be implemented by subclass")


    def get_stations(self):
        """
        Extracts station information from metadata
        """
        raise NotImplementedError("Must be implemented by subclass")


    # the remaining methods deal with indexing and iteration over the obspy
    # streams that comprise the dataset
    def __add__(self, stream):
        assert hasattr(stream, 'id')
        assert isinstance(stream, obspy.Stream)
        stream.tag = 'data'
        self.__list__.append(stream)
        return self


    def remove(self, id):
        index = self._get_index(id)
        self.__list__.pop(index)


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



def reader(*args, **kwargs):
    """
    Each supported file format will have a corresponding reader utility
    that creates an MTUQ Dataset from files stored in that format.  For an
    example, see mtuq.dataset.sac.reader
    """
    pass




import obspy

class DatasetBase(object):
    """ Seismic data container

        Basically, a list of obspy streams. Each stream corresponds to a
        single seismic station and holds all the components recorded at that
        station.  Provides methods to help with data processing and metadata
        extraction.
    """

    def __init__(self, data=None, id=None):
        # event name or other unique identifier
        self.id = id
        self.__list__ = []

        for stream in data:
            self.__add__(stream)


    def apply(self, function, *args, **kwargs):
        """
        Returns the result of applying a function to each Stream in the 
        list. Similar to the behavior of the python built-in "apply".
        """
        processed = self.__class__(id=self.id)
        for stream in self.__list__:
            processed += function(stream, *args, **kwargs)
        return processed


    def map(self, function, *sequences):
        """
        Returns the result of applying a function to each Stream in the
        list. If one or more optional sequences are given, the function is 
        called with an argument list consisting of the corresponding item of
        each sequence. Similar to the behavior of the python built-in "map".
        """
        processed = self.__class__(id=self.id)
        for _i, stream in enumerate(self.__list__):
            args = [sequence[_i] for sequence in sequences]
            processed += function(stream, *args)
        return processed


   # the next two methods are deferred to the subclass, because the way 
   # metadata are organized in obspy streams depends on the file format from 
   # which the stream was read

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


    # the remaining methods deal with indexing and iteration over the list of
    # streams that comprises the dataset

    def __add__(self, stream):
        assert hasattr(stream, 'id')
        assert isinstance(stream, obspy.Stream)
        stream.tag = 'data'
        self.__list__.append(stream)
        return self


    def remove(self, id):
        index = self._get_index[id]
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


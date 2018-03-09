
import obspy

class Dataset(object):

    def __init__(self, data=None):
        if not data:
            self.__list__ = []
            return

        for stream in data:
            self.__add__(stream)


    def apply(self, function, *args, **kwargs):
        """
        Returns the result of applying a function to each Stream in the 
        list. Similar to the behavior of the python built-in "apply".
        """
        processed = Dataset()
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
        processed = Dataset()
        for _i, stream in enumerate(self.__list__):
            args = [sequence[_i] for sequence in sequences]
            processed += function(stream, *args)
        return processed


    def __add__(self, stream):
        assert hasattr(stream, 'id')
        assert isinstance(stream, obspy.Stream)
        self.__list__ += [stream]
        return self


    def __iter__(self):
        return self.__list__.__iter__()


    def __getitem__(self, index):
        return self.__list__[index]


    def __setitem__(self, index, value):
        self.__list__[index] = value


    def __len__(self):
        return len(self.__list__)



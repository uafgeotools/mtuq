
def Dataset(object):
    """ 
    A list of obspy Streams
    """
    def __init__(self, stream):
        """ 
        """
        # sort by station
        data_dict = {}
        for trace in stream:
            id = _id(trace.stats)
            if id not in data_dict:
                data_dict[id] = Stream(trace)
            else:
                data_dict[id] += trace

        self.__list__ = data_dict.values()


    def apply(self, function, *args, **kwargs):
        """
        Returns the result of applying a function to each Stream in the 
        list. Similar to the behavior of the python built-in "apply".
        """
        processed = Dataset()
        for stream in self.__list__:
            processed +=\
                stream.apply(function, *args, **kwargs)
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
            processed +=\
                stream.apply(function, *args)
        return processed


    def __add__(self, stream):
        self.__list__ += [stream]
        return self


    def __iter__(self):
        return self.__list__.__iter__()


    def __getitem__(self, index):
        return self.__list__[index]


    def __setitem__(self, index, value):
        self.__list__[index] = value


    @property
    def stations(self):
        stations = []
        for stream in self.__list__:
            stations += [stream.station]
        return stations




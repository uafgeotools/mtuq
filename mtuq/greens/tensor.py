

class GreensTensor(object):
    """
    Elastic Green's tensor object.  Similar to an obpy Trace, except rather than
    a single time series, a holds six time series corresponding to the 
    the indepdent elements of an elastic tensor

    param tensor:  Green's tensor
    type tensor: list of numpy arrays
    param stats: dictionary containing station and event information
    type stats: obspy Stats dictionary
    """
    def __init__(self, tensor, stats=None):
        self.tensor = tensor
        self.stats = stats


    def combine(self, mt):
        """
        Given a moment tensor, generates synthetics by combining elements of
        the Green's tensor and rotating to given orientation code
        """
        tensor = self.tensor
        stats = self.stats

        data = np.zeros(stats.nsamples)
        for _i in range(3):
            index = _index_map[(_i, component)]
            data += mt[index] * tensor[index]

        return obspy.core.stream.Stream(
            obspy.core.trace.Trace(data, stats))


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all six Green's functions
        """
        tensor = deepcopy(self.tensor)
        stats = deepcopy(self.stats)

        for time_series in tensor:
            time_sereis = function(time_series, *args, **kwargs)
        
        return Tensor(data, headers=meta)


    def convolve(self, wavelet):
        """
        Convolves Green's functions with a source wavelet
        """
        return self.process(np.convolve, wavelet)
        


class GreensTensorList(object):
    """ 
    A list of GreensTensors.  Similar to an obspy Stream, except rather than 
    traces, holds elastic Green's tensors
    """
    def __init__(self):
        self.tensors = []
        self._stations = []
        self._channels = []


    def combine(self, mt):
        stream = obspy.core.stream.Stream()
        for tensor in self.tensors:
            stream += tensor.combine()


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all tensors in list
        """
        processed_tensors = GreensTensorList()
        for tensor in self.tensors:
            processed_tensors += tensor.process(function, *args, **kwargs)
        return processed_tensors


    def convolve(self, wavelet):
        """ 
        Convolves all tensors with given wavelet
        """
        convolved_tensors = GreensTensorList()
        for tensor in self.tensors:
            convolved_tensors += tensor.convolve(wavelet)
        return convolved_tensors


    def __iter__(self):
        return self.tensors.__iter__()


    @property
    def channels:
       if len(self._channels) != len(self.tensor)
           self._channels = [tensor.stats.id for tensor in self.tensors]
       return self._channels

    @property
    def stations:
       if len(self._stations) != len(self.tensor)
           self._stations = [tensor.stats.station for tensor in self.tensors]
       return self._stations




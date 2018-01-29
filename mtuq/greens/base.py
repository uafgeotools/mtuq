
import obspy


class GreensTensor(object):
    """
    Elastic Green's tensor object.  Similar to an obpy Trace, except rather than
    a single time series, a multiple time series corresponding to the 
    the indepdent elements of an elastic tensor

    param tensor:  Green's tensor
    type tensor: list of numpy arrays
    param stats: dictionary containing station and event information
    type stats: obspy Stats dictionary
    """
    def __init__(self, tensor, starttime, sampling_rate, station):
        self.data = data
        self.stats = Stats(
            network=station.network,
            station=station.station,
            location=station.location,
            latitude=station.latitude,
            longitude=station.longitude,
            azimuth=statin.azimuth,
            back_azimuth=station.back_azimuth)


    def combine(self, mt):
        """
        Given a moment tensor, generates synthetics by combining elements of
        the Green's tensor
        """
        raise NotImplementedError("Must be implemented by subclass")


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's tensor elements
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
        self._list = []


    def __add__(self, new_list):
        self._list +=  new_list


    def combine(self, mt):
        stream = obspy.core.stream.Stream()
        for item in self._list:
            stream += item.combine(mt)


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all tensors in list
        """
        processed_tensors = GreensTensorList()
        for tensor in self._list:
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



class Stats(obspy.core.trace.Stats):
    """
    A container for information about a GreensTensor
    """
    pass



import obspy

from copy import deepcopy


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
    def __init__(self, data, station):
        self.data = data
        self.station = station


    def combine(self, mt):
        """
        Generates synthetic seismogram via linear combination of Green's tensor
        elements
        """
        raise NotImplementedError("Must be implemented by subclass")


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's tensor elements
        """
        raise NotImplementedError("Must be implemented by subclass")


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


    def __add__(self, greens_tensor):
        self._list += [greens_tensor]
        return self


    def __iter__(self):
        return self._list.__iter__()


    @property
    def stations(self):
        _stations = []
        for greens_tensor in self._list:
            _stations += [greens_tensor.station]
        return _stations


    def combine(self, mt):
        synthetics = []
        for greens_tensor in self._list:
            synthetics += [greens_tensor.combine(mt)]
        return synthetics


    def process(self, function, *args, **kwargs):
        """
        Applies a signal processing function to all Green's tensors
        """
        processed = GreensTensorList()
        for greens_tensor in self._list:
            processed +=\
                greens_tensor.process(function, *args, **kwargs)
        return processed


    def convolve(self, wavelet):
        """ 
        Convolves all Green's tensors with given wavelet
        """
        convolved = GreensTensorList()
        for greens_tensor in self._list:
            convolved += greens_tensor.convolve(wavelet)
        return convolved


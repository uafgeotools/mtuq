
import obspy
import numpy as np

from obspy.core import Stream
from mtuq.greens_tensor.syngine import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from mtuq.util.syngine import download_unzip_mt_response, download_force_response,\
     resolve_model,\
     GREENS_TENSOR_FILENAMES, SYNTHETICS_FILENAMES



class Client(ClientBase):
    """ Syngine web service client
        
    .. rubric:: Usage

    To instantiate a syngine client, supply the name of an Earth model from one
    one of the available Earth models listed at 
    http://ds.iris.edu/ds/products/syngine/#models
        
    .. code::

        from mtuq.io.clients.syngine import Client
        db = Client(model=model)

    Then the client can be used to download GreensTensors:

    .. code::
        
        greens_tensors = db.get_greens_tensors(stations, origin)
            

    .. note::

        Syngine is an webservice that provides Green's functions and synthetic
        seismograms for download as compressed SAC files. 

    """

    def __init__(self, path_or_url=None, model=None, 
                 include_mt=True, include_force=False):

        if not path_or_url:
            path_or_url = 'http://service.iris.edu/irisws/syngine/1'
        self.url = path_or_url

        # Checks against list of currently supported models. If necessary,
        # appends required period band suffix
        self.model = resolve_model(model)

        self.include_mt = include_mt
        self.include_force = include_force


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Downloads Green's tensors

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)

        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        stream = Stream()

        # read time series
        stream = Stream()

        if self.include_mt:
            dirname = download_unzip_mt_response(
                self.url, self.model, station, origin)

            for filename in GREENS_TENSOR_FILENAMES:
                stream += obspy.read(dirname+'/'+filename, format='sac')

        if self.include_force:
            filenames = download_force_response(
                self.url, self.model, station, origin)

            dirnames = []
            for filename in filenames:
                dirnames += [unzip(filename)]

            for _i, dirname in enumerate(dirnames):
                for filename in SYNTHETICS_FILENAMES:
                    stream += obspy.read(dirname+'/'+filename, format='sac')

                    # overwrite channel attribute
                    stream[-1].stats.channel = stream[-1].stats.channel[-1]+str(_i)


        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # what are the start and end times of the Green's function?
        t1_old = float(stream[0].stats.starttime)
        t2_old = float(stream[0].stats.endtime)
        dt_old = float(stream[0].stats.delta)

        for trace in stream:
            trace.stats._component = trace.stats.channel[0]

            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old,
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            trace.stats.npts = len(data_new)

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'syngine',
             ]

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)



def download_greens_tensors(stations=[], origins=[], model='', verbose=False, **kwargs):
    """ Downloads Green's tensors from syngine

    Downloads Green's functions for all combinations of stations and origins
    using syngine web service (http://ds.iris.edu/ds/products/syngine/).
    Returns an `mtuq.GreensTensorList` of length `len(stations)*len(origins)`.


    .. rubric :: Input arguments


    ``stations`` (list of `mtuq.Station` objects):
    Stations for which Green's functions will be downloaded
    

    ``origins`` (list of `mtuq.Origin` objects):
    Origins for which Green's functions will be downloaded


    ``model`` (str):
    Earth model for which Green's functions will be downloaded, from list of
    available models given at http://ds.iris.edu/ds/products/syngine/.

    """
    client = Client(model=model, **kwargs)
    return client.get_greens_tensors(stations, origins, verbose=verbose)




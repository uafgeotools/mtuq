
import obspy
import numpy as np

from obspy.core import Stream
from mtuq.greens_tensor.syngine import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from mtuq.util.syngine import download_greens_tensor, download_force_response,\
     resolve_model,\
     GREENS_TENSOR_FILENAMES, SYNTHETICS_FILENAMES
from mtuq.util import unzip



class Client(ClientBase):
    """ 
    Interface to syngine Green's function web service

    .. code:

        db = mtuq.greens.open_db(url, format='syngine')

        greens_tensors = db.read(stations, origin)

    In the first step, the user supplies one of the available Earth models
    listed at http://ds.iris.edu/ds/products/syngine/#models

    In the second step, the user supplies a list of stations and the origin
    locations and times. GreensTensors are then created for all the
    corresponding station-origin pairs.

    .. note:

        Syngine is an webservice that provides Green's functions and synthetic
        seismograms for download as compressed SAC files. 

    """

    def __init__(self, path_or_url=None, model=None, enable_force=False):
        if not path_or_url:
            path_or_url = 'http://service.iris.edu/irisws/syngine/1'
        self.url = path_or_url

        # Checks against list of currently supported models. If necessary,
        # appends required period band suffix
        self.model = resolve_model(model)

        # Moment tensor response will always be downloaded. Optionally, force
        # response can be downloaded as well
        self.enable_force = enable_force


    def _get_greens_tensor(self, station=None, origin=None):
        stream = Stream()

        # download and unzip data
        dirname = unzip(
            download_greens_tensor(self.url, self.model, station, origin))

        # read data
        stream = Stream()
        stream.id = station.id
        for filename in GREENS_TENSOR_FILENAMES:
            stream += obspy.read(dirname+'/'+filename, format='sac')

        if self.enable_force:
            filenames = download_force_response(
                self.url, self.model, station, origin)

            dirnames = []
            for filename in filenames:
                dirnames += [unzip(filename)]

            for _i, dirname in enumerate(dirnames):
                for filename in SYNTHETICS_FILENAMES:
                    stream += obspy.read(dirname+'/'+filename, format='sac')

                    # overwrite channel name
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
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old,
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            trace.stats.npts = len(data_new)

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin, enable_force=self.enable_force)



def get_greens_tensors(stations=[], origins=[], **kwargs):
    client = Client(**kwargs)
    return client.get_greens_tensors(stations, origins)




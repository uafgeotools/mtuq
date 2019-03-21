
import obspy
import numpy as np

from obspy.core import Stream
from mtuq.io.greens_tensor.axisem_netcdf import GreensTensor as GreensTensorBase
from mtuq.io.greens_tensor.base import Client as ClientBase
from mtuq.util.signal import resample
from mtuq.util.syngine import download_greens_tensor, download_force_response,\
     GREENS_TENSOR_FILENAMES, SYNTHETICS_FILENAMES
from mtuq.util.util import unzip


class GreensTensor(GreensTensorBase):
    """
    Adds syngine capabilities to AxiSEM base class

    Syngine is an webservice that provides Green's functions and synthetic
    seismograms for download as compressed SAC files. 

    Syngine uses precomputed AxiSEM databases under the hood, so Green's 
    function conventions, moment tensor conventions, and so on are very similar to
    AxiSEM, and it is not necessary to modify any of the machinery for 
    generating synthetics.
    """
    pass


class Client(ClientBase):
    """ 
    Interface to syngine Green's function web service

    Generates GreenTensorLists via a two-step procedure
        1) db = mtuq.greens.open_db(model=model, format='syngine')
        2) greens_tensors = db.read(stations, origin)

    In the first step, the user supplies one of the available Earth model's
    listed at http://ds.iris.edu/ds/products/syngine/#models

    In the second step, the user supplies a list of stations and the origin
    location and time of an event. GreensTensors are then created for all the
    corresponding station-event pairs.
    """

    def __init__(self, model=None, enable_force=False):

        if not model:
            raise ValueError

        self.model = model
        self.enable_force = enable_force


    def _get_greens_tensor(self, station=None, origin=None):
        stream = Stream()

        # download and unzip data
        dirname = unzip(download_greens_tensor(self.model, station, origin))

        # read data
        stream = Stream()
        stream.id = station.id
        for filename in GREENS_TENSOR_FILENAMES:
            stream += obspy.read(dirname+'/'+filename, format='sac')

        if self.enable_force:
            filenames = download_force_response(self.model, station, origin)
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



def get_greens_tensors(stations=None, origin=None, **kwargs):
    client = Client(**kwargs)
    return client.get_greens_tensors(stations=stations, origin=origin)




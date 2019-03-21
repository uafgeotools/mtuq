
import obspy
import numpy as np
import re

from os.path import exists
from obspy.core import Stream, Trace
from mtuq.io.greens_tensor.axisem_netcdf import GreensTensor as GreensTensorBase
from mtuq.io.greens_tensor.base import Client as ClientBase
from mtuq.util.signal import resample
from mtuq.util.util import path_mtuq, unzip, url2uuid, urlopen_with_retry


GREENS_TENSOR_FILENAMES = [
    'greensfunction_XX.GF001..ZSS.sac',
    'greensfunction_XX.GF001..ZDS.sac',
    'greensfunction_XX.GF001..ZDD.sac',
    'greensfunction_XX.GF001..ZEP.sac',
    'greensfunction_XX.GF001..RSS.sac',
    'greensfunction_XX.GF001..RDS.sac',
    'greensfunction_XX.GF001..RDD.sac',
    'greensfunction_XX.GF001..REP.sac',
    'greensfunction_XX.GF001..TSS.sac',
    'greensfunction_XX.GF001..TDS.sac',
    ]

SYNTHETICS_FILENAMES = [
    'XX.S0001.SE.BXZ.sac',
    'XX.S0001.SE.BXR.sac',
    'XX.S0001.SE.BXT.sac',
    ]


class GreensTensor(GreensTensorBase):
    """
    Adds syngine capabilities to AxiSEM base class

    Syngine is an webservice that provides Green's functions and synthetic
    seismograms for download as compressed SAC files. 

    Syngine uses precomputed AxiSEM databases under the hood, so Green's 
    function conventions, moment tensor conventions, and so on are very similar to
    AxiSEM, and it is not necessary to modify any of the machinery for 
    generating synthetics (except for one possible sign discrepacny).
    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        if self.enable_force:
            raise NotImplementedError(
                "Force source implementation not usuable due to suspected "
                "syngine bugs")



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

        print 't1_old', t1_old
        print 't1_new', t1_new

        print 't2_old', t2_old
        print 't2_new', t2_new

        print 'nt_old', (t2_old-t1_old)/dt_old
        print 'nt_new', (t2_new-t1_new)/dt_new

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



def download_greens_tensor(model, station, origin):
    """ Downloads Green's functions through syngine URL interface
    """
    if hasattr(station, 'distance_in_m'):
        distance_in_deg = _in_deg(station.distance_in_m)
    else:
        distance_in_deg = _in_deg(station.preliminary_distance_in_m)


    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&dt='+str(station.delta)
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance_in_deg)
         +'&sourcedepthinmeters='+str(int(round(origin.depth_in_m)))
         +'&origintime='+str(origin.time)[:-1]
         +'&starttime='+str(origin.time)[:-1])
    filename = (path_mtuq()+'/'+'data/greens_tensor/syngine/cache/'
         +str(url2uuid(url)))
    if not exists(filename):
        print ' Downloading Green''s functions for station %s' % station.station
        urlopen_with_retry(url, filename+'.zip')
    return filename


def download_synthetics(model, station, origin, source):
    """ Downloads synthetics through syngine URL interface
    """
    if len(source)==6:
        args='&sourcemomenttensor='+re.sub('\+','',",".join(map(str, source)))
    elif len(source)==3:
        args='&sourceforce='+re.sub('\+','',",".join(map(str, source)))
    else:
        raise TypeError

    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&dt='+str(station.delta)
         +'&components=ZRT'
         +'&receiverlatitude='+str(station.latitude)
         +'&receiverlongitude='+str(station.longitude)
         +'&sourcelatitude='+str(origin.latitude)
         +'&sourcelongitude='+str(origin.longitude)
         +'&sourcedepthinmeters='+str(int(round(origin.depth_in_m)))
         +'&origintime='+str(origin.time)[:-1]
         +'&starttime='+str(origin.time)[:-1])

    if len(source)==6:
        url+='&sourcemomenttensor='+re.sub('\+','',",".join(map(str, source)))
    elif len(source)==3:
        url+='&sourceforce='+re.sub('\+','',",".join(map(str, source)))
    else:
        raise TypeError

    filename = (path_mtuq()+'/'+'data/greens_tensor/syngine/cache/'
         +str(url2uuid(url)))
    if not exists(filename):
        print ' Downloading waveforms for station %s' % station.station
        urlopen_with_retry(url, filename+'.zip')
    return filename+'.zip'


def download_force_response(model, station, origin):
    forces = []
    forces += [np.array([1., 0., 0.])]
    forces += [np.array([0., 1., 0.])]
    forces += [np.array([0., 0., 1.])]

    filenames = []
    for force in forces:
        filenames += [download_synthetics(model, station, origin, force)]
    return filenames


def get_synthetics_syngine(model, station, origin, mt):
    from mtuq.dataset.sac import read

    dirname = unzip(download_synthetics(model, station, origin, mt))
    stream = read(dirname)[0]

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

        setattr(trace, 'network', station.network)
        setattr(trace, 'station', station.station)
        setattr(trace, 'location', station.location)

    synthetics = Stream()
    for trace in stream:
        component = trace.stats.channel[-1].upper()
        trace = stream.select(component=component)[0]
        synthetics += trace

    return synthetics


def get_greens_tensors(stations=None, origin=None, **kwargs):
    client = Client(**kwargs)
    return client.get_greens_tensors(stations=stations, origin=origin)


def _in_deg(distance_in_m):
    from obspy.geodetics import kilometers2degrees
    return kilometers2degrees(distance_in_m/1000., radius=6371.)



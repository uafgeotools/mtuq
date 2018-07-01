
import instaseis
import obspy
import numpy as np
import re
import urllib
import mtuq.greens_tensor.base
import mtuq.greens_tensor.instaseis

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists
from obspy.core import Stream, Trace
from mtuq.util.geodetics import km2deg
from mtuq.util.signal import resample
from mtuq.util.util import root, unzip, url2uuid


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


class GreensTensor(mtuq.greens_tensor.instaseis.GreensTensor):
    # same as parent class
    pass



class GreensTensorFactory(mtuq.greens_tensor.base.GreensTensorFactory):
    def __init__(self, model):
        self.model = model


    def get_greens_tensor(self, station, origin):
        # download and unizp data
        dirname = unzip(download_greens_tensor(self.model, station, origin))

        # read data
        stream = Stream()
        stream.id = station.id
        for filename in GREENS_TENSOR_FILENAMES:
            stream += obspy.read(dirname+'/'+filename, format='sac')

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
            data_old = trace.data*1.e20/3.2e5 # ADHOC
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            trace.stats.npts = len(data_new)

        traces = [trace for trace in stream]
        return GreensTensor(traces, station, origin)


def download_greens_tensor(model, station, origin):
    """ Downloads Green's functions through syngine URL interface
    """
    try:
        distance_in_deg = km2deg(station.distance)
    except:
        distance_in_deg = km2deg(station.catalog_distance)
    depth_in_m = origin.depth

    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&dt='+str(station.delta)
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance_in_deg)
         +'&sourcedepthinmeters='+str(int(round(depth_in_m)))
         +'&origintime='+str(origin.time)[:-1]
         +'&starttime='+str(origin.time)[:-1])
    filename = (root()+'/'+'data/greens_tensor/syngine/cache/'
         +str(url2uuid(url))
         +'.zip')
    if not exists(filename):
        print ' Downloading waveforms for station %s' % station.station
        download = urllib.URLopener()
        download.retrieve(url, filename)
    return filename


def download_synthetics(model, station, origin, mt):
    """ Downloads synthetics through syngine URL interface
    """
    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&dt='+str(station.delta)
         +'&components=ZRT'
         +'&receiverlatitude='+str(station.latitude)
         +'&receiverlongitude='+str(station.longitude)
         +'&sourcelatitude='+str(origin.latitude)
         +'&sourcelongitude='+str(origin.longitude)
         +'&sourcedepthinmeters='+str(int(round(origin.depth)))
         +'&origintime='+str(origin.time)[:-1]
         +'&starttime='+str(origin.time)[:-1]
         +'&sourcemomenttensor='+re.sub('\+','',",".join(map(str, mt))))
    filename = (root()+'/'+'data/greens_tensor/syngine/cache/'
         +str(url2uuid(url))
         +'.zip')
    if not exists(filename):
        print ' Downloading waveforms for station %s' % station.station
        download = urllib.URLopener()
        download.retrieve(url, filename)
    return filename



def get_synthetics_syngine(model, station, origin, mt):
    from mtuq.data.sac import reader

    dirname = unzip(download_synthetics(model, station, origin, mt))
    stream = reader(dirname)[0]

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
        data_old = trace.data*1.e20/3.2e5 # ADHOC
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
    for channel in station.channels:
        component = channel[-1].upper()
        trace = stream.select(component=component)[0]
        synthetics += trace

    return synthetics


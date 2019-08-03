
import numpy as np

from os.path import exists
from obspy.core import Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from mtuq.util import fullpath, unzip, url2uuid, urlopen_with_retry
from mtuq.util.signal import get_distance_in_m, get_distance_in_deg


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

SYNGINE_MODELS = [
    'ak135f_2s',
    'ak135f_5s',
    'iasp91_2s',
    'prem_i_2s',
    'prem_a_2s',
    'prem_a_5s',
    'prem_a_10s',
    'prem_a_20s',
    ]

def resolve_model(name):
    if not name:
        raise ValueError('Bad model')

    if name=='ak135':
        # temporary workaround, because obspy lacks ak135f
        name+='f'

    if name in SYNGINE_MODELS:
        return name

    name+='_2s'
    if name in SYNGINE_MODELS:
        return name
    else:
        raise ValueError('Bad model')


def download_greens_tensor(url, model, station, origin):
    """ Downloads Green's functions through syngine URL interface
    """
    distance_in_deg = get_distance_in_deg(station, origin)

    url = (url+'/'+'query'
         +'?model='+model
         +'&dt='+str(station.delta)
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance_in_deg)
         +'&sourcedepthinmeters='+str(int(round(origin.depth_in_m)))
         +'&origintime='+str(origin.time)[:-1]
         +'&starttime='+str(origin.time)[:-1])

    filename = fullpath('data/greens_tensor/syngine/cache/', str(url2uuid(url)))
    if exists(filename):
        return filename
    elif exists(filename+'.zip'):
        return filename
    else:
        print ' Downloading Green''s functions for station %s' % station.station
        urlopen_with_retry(url, filename+'.zip')
        return filename


def download_synthetics(url, model, station, origin, source):
    """ Downloads synthetics through syngine URL interface
    """
    if len(source)==6:
        args='&sourcemomenttensor='+re.sub('\+','',",".join(map(str, source)))
    elif len(source)==3:
        args='&sourceforce='+re.sub('\+','',",".join(map(str, source)))
    else:
        raise TypeError

    url = (url+'/'+'query'
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

    filename = fullpath('data/greens_tensor/syngine/cache/', str(url2uuid(url)))
    if exists(filename):
        return filename
    elif exists(filename+'.zip'):
        return filename
    else:
        print ' Downloading waveforms for station %s' % station.station
        urlopen_with_retry(url, filename+'.zip')
        return filename+'.zip'


def download_force_response(url, model, station, origin):
    forces = []
    forces += [np.array([1., 0., 0.])]
    forces += [np.array([0., 1., 0.])]
    forces += [np.array([0., 0., 1.])]

    filenames = []
    for force in forces:
        filenames += [download_synthetics(url, model, station, origin, force)]
    return filenames


def get_synthetics_syngine(url, model, station, origin, mt):
    from mtuq.dataset.sac import read

    dirname = unzip(download_synthetics(url, model, station, origin, mt))
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




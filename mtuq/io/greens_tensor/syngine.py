
import obspy
import numpy as np
import re
import mtuq.io.greens_tensor.axisem_netcdf

from os.path import basename, exists
from obspy.core import Stream, Trace
from mtuq.util.signal import resample
from mtuq.util.util import path_mtuq, unzip, url2uuid, urlopen_with_retry


# syngine is an webservice that provides Green's functions and synthetic
# seismograms for download as compressed SAC files. syngine uses instaseis
# under the hood for preparing user downloads, so Green's function conventions,
# moment tensor conventions, metadata format, data processing, and so on are
# very similar for syngine and instaseis.  In MTUQ, all of the machinery for 
# generating synthetics from syngine Green's tensors is inherited from 
# mtuq.greens_tensor.instaseis


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


class GreensTensor(mtuq.io.greens_tensor.axisem_netcdf.GreensTensor):
    def _precompute_weights(self):
        super(GreensTensor, self)._precompute_weights()

        # the negative sign is needed because of a bug in syngine? or because 
        # of inconsistent moment tensor conventions?
        self._weighted_tensor[2] *= -1


        # Order of terms expected by syngine URL parser (from online
        # documentation):
        #    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
        #
        # Relations given in instaseis/tests/test_instaseis.py:
        #    m_tt=Mxx, m_pp=Myy, m_rr=Mzz, m_rt=Mxz, m_rp=Myz, m_tp=Mxy
        #
        # Relations suggested by mtuq/tests/unittest_greens_tensor_syngine.py
        # (note sign differences):
        #    m_tt=Mxx, m_pp=Myy, m_rr=Mzz, m_rt=-Mxz, m_rp=Myz, m_tp=-Mxy


    def get_synthetics(self, source):
        if len(source)==6:
            return self._get_mt_synthetics(source)

        elif len(source)==3:
            return self._get_force_synthetics(source)


    def _get_force_synthetics(self, force):
        for _i, component in enumerate(self.components):
            # which Green's functions correspond to given component?
            if component=='Z':
                _j=0
            elif component=='R':
                _j=1
            elif component=='T':
                _j=2
            G = self._weighted_tensor[_j]

            # we could use np.dot instead, but speedup appears negligible
            s = self._synthetics[_i].data
            s[:] = 0.
            s += force[0]*G[:,0]
            s += force[1]*G[:,1]
            s += force[2]*G[:,2]


    def _get_mt_synthetics(self, mt):
        return super(GreensTensor, self).get_synthetics(mt)


class Client(mtuq.io.greens_tensor.axisem_netcdf.Client):
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

    def __init__(self, model=None, 
            include_mt_response=True, include_force_response=True):

        self.include_mt_response = include_mt_response
        self.include_force_response = include_force_response

        if not model:
            raise ValueError
        self.model = model


    def _get_greens_tensor(self, station, origin):
        traces = []

        if self.include_force_response:
            traces += self._get_force_response(station, origin)

        if self.include_mt_response:
            traces += self._get_mt_response(station, origin) 

        return GreensTensor(traces, station, origin)


    def _get_mt_response(self, station, origin):
        # download and unzip data
        dirname = unzip(download_greens_tensor(self.model, station, origin))

        # read data
        stream = Stream()
        stream.id = station.id
        for filename in GREENS_TENSOR_FILENAMES:
            stream += obspy.read(dirname+'/'+filename, format='sac')

        return self._resample(stream, station)


    def _get_force_response(self, station, origin):
        # download and unzip data
        filenames = download_force_response(self.model, station, origin)
        dirnames = []
        for filename in filenames:
            dirnames += [unzip(filename)]

        # read data
        stream = Stream()
        stream.id = station.id

        for _i, dirname in enumerate(dirnames):
            for filename in SYNTHETICS_FILENAMES:
                stream += obspy.read(dirname+'/'+filename, format='sac')

                # overwrite channel name
                stream[-1].stats.channel = str(_i)+stream[-1].stats.channel[-1]
                print stream[-1].stats.channel

        return self._resample(stream, station)


    def _resample(self, stream, stats):
        # what are the start and end times of the data?
        t1_new = float(stats.starttime)
        t2_new = float(stats.endtime)
        dt_new = float(stats.delta)

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

        return [trace for trace in stream]


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
         +'&starttime='+str(origin.time)[:-1]
         +_syngine_source_args(source))
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


def _in_deg(distance_in_m):
    from obspy.geodetics import kilometers2degrees
    return kilometers2degrees(distance_in_m/1000., radius=6371.)


def _syngine_source_args(source):
    if len(source)==6:
        return '&sourcemomenttensor='+re.sub('\+','',",".join(map(str, source)))
    elif len(source)==3:
        return '&sourceforce='+re.sub('\+','',",".join(map(str, source)))
    else:
        raise TypeError


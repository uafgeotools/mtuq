
import instaseis
import obspy
import numpy as np
import urllib

import mtuq.greens_tensor.base
import mtuq.greens_tensor.instaseis

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists

from obspy.core import Stream, Trace
from mtuq.util.geodetics import km2deg
from mtuq.util.signal import resample
from mtuq.util.util import unzip


SYNGINE_FILENAMES = [
    'greensfunction_XX.GF001..RDD.sac',
    'greensfunction_XX.GF001..RDS.sac',
    'greensfunction_XX.GF001..REP.sac',
    'greensfunction_XX.GF001..RSS.sac',
    'greensfunction_XX.GF001..TDS.sac',
    'greensfunction_XX.GF001..TSS.sac',
    'greensfunction_XX.GF001..ZDD.sac',
    'greensfunction_XX.GF001..ZDS.sac',
    'greensfunction_XX.GF001..ZEP.sac',
    'greensfunction_XX.GF001..ZSS.sac',
    ]

# If a GreensTensor is created with the wrong input arguments, this error
# message is displayed.  In practice this is rarely encountered, since
# GreensTensorFactory normally does all the work
ErrorMessage=''


class GreensTensor(mtuq.greens_tensor.instaseis.GreensTensor):
    # same as parent class
    pass



class GreensTensorFactory(mtuq.greens_tensor.base.GreensTensorFactory):
    def __init__(self, model):
        self.model = model


    def get_greens_tensor(self, station, origin):
        # arguments list
        model = self.model
        delta = station.delta
        distance = km2deg(station.distance)
        depth = origin.depth

        # download and unizp data
        dirname = unzip(download_greens_functions(model, delta, distance, depth))

        # read data
        stream = Stream()
        stream.id = station.id
        for filename in SYNGINE_FILENAMES:
            stream += obspy.read(dirname+'/'+filename, format='sac')

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # what are the start and end times of the Green's function?
        t1_old = float(origin.time)+float(stream[0].stats.starttime)
        t2_old = float(origin.time)+float(stream[0].stats.endtime)
        dt_old = float(stream[0].stats.delta)

        print(t1_old, t2_old, dt_old, t1_new, t2_new, dt_new)
            

        for trace in stream:
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

        return GreensTensor(stream, station, origin)


def download_greens_functions(model, delta, distance_in_deg, depth_in_m):
    """ Downloads Green's functions through syngine URL interface
    """
    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&dt='+str(delta)
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance_in_deg)
         +'&sourcedepthinmeters='+str(int(round(depth_in_m))))
    filename = ('tmp-'
         +'model='+model
         +'dt='+str(delta)
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance_in_deg)
         +'&sourcedepthinmeters='+str(int(round(depth_in_m)))
         +'.zip')
    download = urllib.URLopener()
    download.retrieve(url, filename)
    return filename


def download_synthetics(mt, model, delta, station, origin):
    """ Downloads synthetics through syngine URL interface
    """
    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&dt='+str(delta)
         +'&receiverlatitude'+str(station.latitude)
         +'&receiverlongitude'+str(station.longitude)
         +'&sourclatitude'+str(origin.latitude)
         +'&sourcelongitude'+str(origin.longitude)
         +'&sourcedepthinmeters='+str(int(round(origin.depth)))
         +'&sourcemomenttensor='+",".join(map(str, mt)))
    filename = ('tmp-'
         +'model='+model
         +'dt='+str(delta)
         +'&receiverlatitude'+str(station.latitude)
         +'&receiverlongitude'+str(station.longitude)
         +'&sourclatitude'+str(origin.latitude)
         +'&sourcelongitude'+str(origin.longitude)
         +'&sourcedepthinmeters='+str(int(round(origin.depth)))
         +'&sourcemomenttensor='+",".join(map(str, mt))
         +'.zip')
    download = urllib.URLopener()
    download.retrieve(url, filename)
    return filename


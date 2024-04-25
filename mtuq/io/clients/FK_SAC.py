
import obspy
import os
import numpy as np

from glob import glob
from os.path import basename, exists, isdir
from os import listdir
from mtuq.greens_tensor.FK import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth


# An FK simulation outputs 12 SAC files each with filename extensions
# 0,1,2,3,4,5,6,7,8,9,a,b.  The SAC files ending in .2 and .9 contain 
# only zero data, so we exclude them from the following list. 
# The order of traces in the list is the order in which CAP stores
# the time series.
EXTENSIONS = [
    '8','5',           # t
    'b','7','4','1',   # r
    'a','6','3','0',   # z
    ]

CHANNELS = [
    'TSS', 'TDS',
    'REP', 'RSS', 'RDS', 'RDD',
    'ZEP', 'ZSS', 'ZDS', 'ZDD',
    ]




class Client(ClientBase):
    """  FK database client

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.FK_SAC import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

      `GreensTensor`s are obtained by reading precomputed time series from an 
      FK directory tree.  Such trees contain SAC files organized by model, 
      event depth, and event distance, as used by the `Zhu1994`
      software packages.

    """
    def __init__(self, path_or_url=None, model=None,
        include_mt=True, include_force=False):

        if not path_or_url:
            raise Exception

        if not exists(path_or_url):
            raise Exception

        if include_force:
            raise NotImplementedError

        if not model:
            model = basename(path_or_url)

        # path to fk directory tree
        self.path = path_or_url

        # model from which fk Green's functions were computed
        self.model = model

        self.include_mt = include_mt
        self.include_force = include_force



    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Extracts Green's tensors from database

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)

        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        if station is None:
            raise Exception("Missing station input argument")

        if origin is None:
            raise Exception("Missing station input argument")

        traces = []

        distance_in_m, _, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        depth_km = _closest_depth(self.path, np.ceil(origin.depth_in_m/1000.))
        offset_km = _closest_offset(self.path, depth_km, distance_in_m/1000.)


        if self.include_mt:

            for _i, ext in enumerate(EXTENSIONS):
                trace = obspy.read('%s/%s_%d/%d.grn.%s' %
                    (self.path, self.model, depth_km, offset_km, ext),
                    format='sac')[0]

                trace.stats.channel = CHANNELS[_i]
                trace.stats._component = CHANNELS[_i][0]


                # what are the start and end times of the Green's function?
                t1_old = float(origin.time)+float(trace.stats.starttime)
                t2_old = float(origin.time)+float(trace.stats.endtime)
                dt_old = float(trace.stats.delta)
                data_old = trace.data

                # resample Green's function
                data_new = resample(data_old, t1_old, t2_old, dt_old, 
                                              t1_new, t2_new, dt_new)
                trace.data = data_new
                # convert from 10^-20 dyne to N^-1
                trace.data *= 1.e-15
                trace.stats.starttime = t1_new
                trace.stats.delta = dt_new

                traces += [trace]

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'FK',
             ]

        return GreensTensor(traces=[trace for trace in traces], 
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)



#
# utility functions
#

def _closest_depth(path, depth_km, thresh_km=1.):
    """ Searches FK directory tree to find closest depth for which 
    Green's functions are available
    """

    if not _listdir(path):
        raise Exception('No subdirectories found: %s' % path)

    depths = []
    for subdir in _listdir(path):
        try:
            # FK naming convention is {model}_{depth}
            parts = subdir.split('_')
            assert len(parts) == 2
            assert parts[1].isdigit()
        except:            
            continue

        # keep track of depths in km
        depths += [float(parts[1])]

    closest_depth = min(depths, key=lambda z: abs(z - depth_km))

    if (closest_depth - depth_km) > thresh_km:
        print('Warning: Closest available Greens functions differ from given source '
              'by %.f km vertically' % (closest_depth - depth_km))
        print('Warning: Depth displayed in figure header may be inaccurate')

    return closest_depth


def _closest_offset(path, depth_km, offset_km, thresh_km=1.):
    """ Searches FK directory tree to find closest horizontal offset for which 
    Green's functions are available
    """

    # the directory naming convention used by FK is {model}_{depth}
    wildcard = '%s/*_%d' % (path, depth_km)

    # the file naming convention used by FK is {offset}.grn.{extension}
    wildcard = '%s/*.grn.0' % wildcard

    if not glob(wildcard):
        raise Exception('No Greens functions found: %s' % wildcard)

    offsets = []
    for fullname in glob(wildcard):
        filename = os.path.basename(fullname)

        # exclude improperly-formatted filenames
        parts = filename.split('.')
        if len(parts) != 3:
            continue
        if not parts[0].isdigit():
            continue

        # keep track of horizontal offsets in km
        offsets += [float(parts[0])]

    closest_offset = min(offsets, key=lambda r: abs(r - offset_km))

    if (closest_offset - offset_km) > thresh_km:
        print('Warning: Closest available Greens functions differ from given source '
              'by %.f km horizontally' % (closest_offset - offset_km))

    return closest_offset


def _listdir(path):
    # lists all subdirectories
    for name in listdir(path):
        if isdir(os.path.abspath(os.path.join(path, name))):
            yield name



import obspy
import os
import numpy as np

from glob import glob
from os.path import basename, exists, isdir, join
from os import listdir
from mtuq.greens_tensor.CPS import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth


# CPS workflow (solver + binary-to-SAC conversion) produces the following
# individual time series
CHANNELS = [
    'ZEX', 'ZSS', 'ZDS', 'ZDD',
    'REX', 'RSS', 'RDS', 'RDD',
    'TSS', 'TDS',
]

class Client(ClientBase):
    """  CPS database client

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.CPS_SAC import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

      `GreensTensor`s are obtained by reading precomputed time series from an 
      CPS directory tree with naming convention `ZZZz/RRRRrZZZz.EXT`,
      where ZZZz gives the depth of the source, RRRRr gives the horizontal offset
      between source and receiver, and EXT is the file extension related to
      so-called fundamental sources.

    .. note::

      The above directory tree convention permits us to represent offsets 
      from 0 to 9999.9 km in 0.1 km increments and origin depths from 
      0 to 999.9 km in 0.1 km increments



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

        # path to CPS directory tree
        self.path = path_or_url

        # model from which CPS Green's functions were computed
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

        offset_in_m, _, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # what are the start and end times of the data?
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        if self.include_mt:
            # closest depth for which Green's functions are available
            depth_km = _closest_depth(self.path, origin.depth_in_m/1000.)

            # closest horizontal offset for which Green's functions are available
            offset_km = _closest_offset(self.path, depth_km, offset_in_m/1000.)

            # the file naming convention used by CPS for offset is RRRRr, which
            # allows us to represent offsets up to 9999.9 km 
            # in increments of 0.1 km
            offset_str = '%05d' % (10.*offset_km)

            # the file naming convention used by CPS for depth is ZZZz, which
            # allows us to represent depths up to 999.9 km 
            # in increments of 0.1 km
            depth_str = '%04d' % (10.*depth_km)


            for _i, ext in enumerate(CHANNELS):
                trace = obspy.read('%s/%s/%s%s.%s' %
                                   (self.path, depth_str,
                                    offset_str, depth_str, ext),
                                   format='sac')[0]

                trace.stats.channel = CHANNELS[_i]
                trace.stats._component = CHANNELS[_i][0]

                # ad hoc workaround for difference between 
                # CPS and FK conventions
                if CHANNELS[_i].endswith('EX'):
                    trace.stats.channel = trace.stats._component+'EP'

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

        if self.include_force:
            raise NotImplementedError

        tags = [
            'model:%s' % self.model,
            'solver:%s' % 'CPS',
        ]

        return GreensTensor(traces=[trace for trace in traces],
                            station=station, origin=origin, tags=tags,
                            include_mt=self.include_mt, include_force=self.include_force)


#
# utility functions
#

def _closest_depth(path, depth_km, thresh=1.):
    """ Searches CPS directory tree to find closest depth for which 
    Green's functions are available
    """
    if not _listdir(path):
        raise Exception('No subdirectories found: %s' % path)

    depths = []
    for subdir in _listdir(path):
        # exclude improperly-formatted subdirectories
        if len(subdir) != 4:
            continue
        if not subdir.isdigit():
            continue

        # convert to km
        depths += [float(subdir)/10.]
      
    closest_depth = min(depths, key=lambda z: abs(z - depth_km))

    if (closest_depth - depth_km) > thresh:
        print('Warning: Closest available Greens functions differ from given source '
              'by %.f km vertically' % (closest_depth - depth_km))

    return closest_depth


def _closest_offset(path, depth_km, offset_km, thresh=1.):
    """ Searches CPS directory tree to find closest horizontal offset for which 
    Green's functions are available
    """

    # the file naming convention used by CPS is ZZZz/RRRRrZZZz.EXT
    wildcard = '%s/%04d/?????%04d.ZEX' % (path, 10*depth_km, 10*depth_km)

    if not glob(wildcard):
        raise Exception('No Greens functions found: %s' % wildcard)

    offsets = []
    for fullname in glob(wildcard):
        filename = os.path.basename(fullname)

        # exclude improperly-formatted filenames
        if not filename[0:9].isdigit():
            continue

        # convert to km
        offsets += [float(filename[0:5])/10.]
      
    closest_offset = min(offsets, key=lambda r: abs(r - offset_km))

    if (closest_offset - offset_km) > thresh:
        print('Warning: Closest available Greens functions differ from given source '
              'by %.f km horizontally' % (closest_offset - offset_km))

    return closest_offset


def _listdir(path):
    # lists all subdirectories
    for name in listdir(path):
        if isdir(os.path.abspath(os.path.join(path, name))):
            yield name
 

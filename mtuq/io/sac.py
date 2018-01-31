
import glob
import os
import warnings
from copy import deepcopy
from os.path import join

import numpy as np
import obspy
from obspy.core import Stream, Stats, Trace
from obspy.core.event import Event, Origin
from obspy.core.inventory import Inventory, Station
from obspy.core.util.attribdict import AttribDict


def read(path, wildcard='*', verbose=False):
    """ Reads SAC files
    """
    event_name = os.path.dirname(path)
    files = glob.glob(join(path, wildcard))

    # read data, one file at a time
    data = obspy.core.Stream()
    for filename in files:
        try:
            data += obspy.read(filename, format='sac')
        except:
            if verbose:
                print('Not a SAC file: %f' % filename)

    # sort by station
    data_dict = {}
    for trace in data:
        id = _id(trace.stats)
        if id not in data_dict:
            data_dict[id] = Stream(trace)
        else:
            data_dict[id] += trace

    data_sorted = data_dict.values()
    return data_sorted


def get_origin(data, event_name=None):
    sac_headers = data[0][0].stats.sac

    # location
    try:
        latitude = sac_headers.evla
        longitude = sac_headers.evlo
    except (TypeError, ValueError):
        warnings.warn("Could not determine event location from sac headers. "
                      "Setting location to nan...")
        latitude = np.nan
        longitudue = np.nan

    # depth
    try:
        depth = sac_headers.evdp
    except (TypeError, ValueError):
        warnings.warn("Could not determine event depth from sac headers. "
                      "Setting depth to nan...")
        depth = 0.

    # origin time
    try:
        origin_time = obspy.UTCDateTime(
            year=sac_headers.nzyear,
            julday=sac_headers.nzjday, 
            hour=sac_headers.nzhour, 
            minute=sac_headers.nzmin,
            second=sac_headers.nzsec) 
    except (TypeError, ValueError):
        warnings.warn("Could not determine origin time from sac headers. "
                      "Setting origin time to zero...")
        origin_time = obspy.UTCDateTime(0)

    return Origin(
        time=origin_time,
        longitude=sac_headers.evlo,
        latitude=sac_headers.evla,
        depth=depth * 1000.0,
    )


def get_stations(data):
    """ Collect station metadata from obspy Stream objects
    """
    stations = []
    for stream in data:
        stats = deepcopy(stream[0].stats)
        try:
            station_latitude = stats.sac.stla
            station_longitude = stats.sac.stlo
        except:
            raise Exception(
                "Could not determine station location from SAC headers.")

        try:
            station_elevation = stats.sac.stel
            station_depth = stats.sac.stdp
        except:
            warnings.warn(
                "Could not determine station elevation, depth from SAC headers.")
            station_elevation = 0.
            station_depth = 0.

        try:
            event_latitude = stats.sac.evla
            event_longitude = stats.sac.evlo
        except:
            raise Exception(
                "Could not determine event location from SAC headers.")

        stats.update({
           'starttime': stats.starttime,
           'delta': stats.delta,
           'latitude':station_latitude,
           'longitude':station_longitude,
           'elevation':station_elevation,
           'depth':station_depth,
           #'distance': distance,
           #'azimuth': azimuth,
           #'back_azimuth': back_azimuth,
           #'event_depth': event_depth,
           })

        stats.channels = []
        for trace in stream:
            stats.channels += [trace.stats.channel]
        stations += [stats]

    return stations


def _id(stats):
    return '.'.join((
        stats.network,
        stats.location,
        stats.station))

def _copy(stats):
    stats = deepcopy(stats)
    stats['channels'] = stats.pop('channel')
    return stats



# debugging
if __name__=='__main__':
    data = read('/u1/uaf/rmodrak/packages/capuaf/20090407201255351')
    get_stations(data)


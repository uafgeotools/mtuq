
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
from obspy.geodetics import gps2dist_azimuth
from mtuq.util.signal import check_time_sampling


def read(path, wildcard='*.sac', verbose=False):
    """ Reads SAC traces and sorts them by station

     Additional processing would be required if for a given station, time
     sampling varies from one channel to another
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
        station_metadata = _copy(stream[0].stats)
      
        if not check_time_sampling(stream):
            # ordinarily we except all traces from a given station to have the 
            # same time sampling
            raise NotImplementedError(
                "Time sampling differs from trace to trace.")

        try:
            station_metadata.channels = []
            for trace in stream:
                station_metadata.channels += [trace.stats.channel]
        except:
            raise Exception(
                "Could not determine channel names from obspy stream.")

        try:
            station_latitude = station_metadata.sac.stla
            station_longitude = station_metadata.sac.stlo
            station_metadata.update({
                'latitude': station_latitude,
                'longitude': station_longitude})
        except:
            raise Exception(
                "Could not determine station location from SAC headers.")

        try:
            station_metadata.update({
                'station_elevation': station_metadata.sac.stel,
                'station_depth': station_metadata.sac.stdp})
        except:
            warnings.warn(
                "Could not determine station elevation, depth from SAC headers.")

        try:
            # if hypocenter is included as an inversion parameter, then we 
            # cannot rely on these sac metadata fields, which are likely based
            # on catalog locations or other preliminary information
            event_latitude = stats.sac.evla
            event_longitude = stats.sac.evlo
            distance, azimuth, backazimuth = obspy.geodetics.gps2dist_azimuth(
                station.latitude,
                station.longitude,
                origin.latitude,
                origin.longitude)

            station_metadata.update({
                'catalog_latitude': event_latitude,
                'catalog_longitude': event_longitude,
                'catalog_distance': distance,
                'catalog_azimuth': azimuth,
                'catalog_backazimuth': back_azimuth})

        except:
            warnings.warn(
                "Could not determine event location from SAC headers.")

        try:
            print stats.sac.t5
            # ordinarily we calculate phase arrival times ourselves using 
            # obspy.taupy; the only reason for checking P arrival times in SAC
            # metadata is to test our code against the legacy CAP package
            station_metadata.update({'arrival_P_sac': stats.sac.t5})
        except:
            # do nothing
            pass

        del station_metadata.sac
        stations += [station_metadata]
    return stations


### utility functions

def _id(stats):
    return '.'.join((
        stats.network,
        stats.location,
        stats.station))

def _copy(stats):
    stats = deepcopy(stats)
    stats.channel = None
    return stats



# debugging
if __name__=='__main__':
    data = read('/u1/uaf/rmodrak/packages/capuaf/20090407201255351')
    get_stations(data)


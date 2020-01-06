
import glob
import os
import numpy as np
import obspy

from os.path import join
from obspy.core import Stream
from mtuq import Dataset, Origin, Station
from mtuq.util import iterable, warn
from mtuq.util.signal import check_components, check_time_sampling


def read(filenames, station_id_list=None, event_id=None, tags=[]):
    """ 
    Reads SAC files and returns MTUQ Dataset

    .. rubric :: Parameters

    - ``filenames`` (``list``)
      List of SAC files to be read (can contain Unix wildcards)

    - ``station_id_list`` (``list``)
      Any traces that are not from one of the listed stations will be excluded

    - ``event_id`` (``str``)
     Identifier to be suppplied to the MTUQ Dataset

    - ``tags`` (``list``)
      Tags to be supplied to the MTUQ Dataset

    """
    # read traces one at a time
    data = Stream()
    for filename in _glob(filenames):
        try:
            data += obspy.read(filename, format='sac')
        except:
            print('Not a SAC file: %d' % filename)

    # sort by station
    data_sorted = {}
    for trace in data:
        station_id = '.'.join((
            trace.stats.network,
            trace.stats.station,
            trace.stats.location))

        if station_id not in data_sorted:
            data_sorted[station_id] = Stream(trace)
        else:
            data_sorted[station_id] += trace

    if station_id_list is not None:
        # remove traces not from station_id_list
        for station_id in data_sorted:
            if station_id not in station_id_list:
                data_sorted.pop(station_id)

    streams = []
    for station_id in data_sorted:
         streams += [data_sorted[station_id]]

    # check for duplicate components
    for stream in streams:
        check_components(stream)

    # collect event metadata
    preliminary_origin = _get_origin(streams[0], event_id)
    for stream in streams:
        assert preliminary_origin==_get_origin(stream, event_id)
        stream.origin = preliminary_origin
    tags += ['origin_type:preliminary']

    # collect station metadata
    for stream in streams:
        stream.station = _get_station(stream, preliminary_origin)

    # create MTUQ Dataset
    return Dataset(streams, id=event_id, tags=tags)


def _get_origin(stream, event_id):
    """ Extracts event metadata from SAC headers

    At the beginning of an inversion, MTUQ requires preliminary estimates for
    event location and depth. We obtain these from SAC headers, which for IRIS
    data represent catalog solutions 
    """
    sac_headers = stream[0].stats.sac

    try:
        latitude = sac_headers.evla
        longitude = sac_headers.evlo
    except (TypeError, ValueError):
        warn("Could not determine event location from sac headers. "
              "Setting location to nan...")
        latitude = np.nan
        longitudue = np.nan

    try:
        depth_in_m = sac_headers.evdp*1000.
    except (TypeError, ValueError):
        warn("Could not determine event depth from sac headers. "
             "Setting depth to nan...")
        depth_in_m = 0.

    try:
        origin_time = obspy.UTCDateTime(
            year=sac_headers.nzyear,
            julday=sac_headers.nzjday, 
            hour=sac_headers.nzhour,
            minute=sac_headers.nzmin,
            second=sac_headers.nzsec)
    except (TypeError, ValueError):
        warn("Could not determine origin time from sac headers. "
              "Setting origin time to zero...")
        origin_time = obspy.UTCDateTime(0)

    return Origin({
        'id': event_id,
        'time': origin_time,
        'longitude': longitude,
        'latitude': latitude,
        'depth_in_m': depth_in_m
        })


def _get_station(stream, origin):
    """ Extracts station metadata from SAC headers
    """
    station = Station(stream[0].meta)
    sac_headers = station.sac

    station.update({
        'id': '.'.join([
            stream[0].stats.network,
            stream[0].stats.station,
            stream[0].stats.location])})

    try:
        station_latitude = sac_headers.stla
        station_longitude = sac_headers.stlo
        station.update({
            'latitude': station_latitude,
            'longitude': station_longitude})
    except:
        raise Exception(
            "Could not determine station location from SAC headers.")

    try:
        station.update({
            'station_elevation_in_m': sac_headers.stel,
            'station_depth_in_m': sac_headers.stdp})
    except:
        pass

    return station


def _glob(filenames):
   # glob any wildcards
   _list = list()
   for filename in iterable(filenames):
       _list.extend(glob.glob(filename))
   return sorted(_list)



import glob
import os
import numpy as np
import obspy

from os.path import join
from obspy.core import Stream
from mtuq import EventDataset, Origin, Station
from mtuq.util import iterable, warn
from mtuq.util.signal import check_time_sampling


def read(filenames, event_id=None, tags=[]):
    """ Creates MTUQ Dataset from SAC traces
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
        id = '.'.join((
            trace.stats.network,
            trace.stats.station,
            trace.stats.location))

        if id not in data_sorted:
            data_sorted[id] = Stream(trace)
        else:
            data_sorted[id] += trace

    streams = []
    for id in data_sorted:
        streams += [data_sorted[id]]

    # collect event metadata
    preliminary_origin = _get_origin(streams[0], event_id)
    for stream in streams:
        assert preliminary_origin==_get_origin(stream, event_id)

    # collect station metadata
    stations = []
    for stream in streams:
        station = _get_station(stream, preliminary_origin)
        stations += [station]

    # create MTUQ Dataset
    return EventDataset(streams=streams, 
        stations=stations, 
        origin=preliminary_origin,
        id=event_id,
        tags=tags)


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

    try:
        station.update({
            'preliminary_event_latitude': origin.latitude,
            'preliminary_event_longitude': origin.longitude,
            'preliminary_event_depth_in_m': origin.depth_in_m})
    except:
        print("Could not determine event location.")

    try:
        station.update({
            'preliminary_origin_time': origin.time})
    except:
        print("Could not determine origin time.")

    return station


def _glob(filenames):
   # glob any wildcards
   _list = list()
   for filename in iterable(filenames):
       _list.extend(glob.glob(filename))
   return _list


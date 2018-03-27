
import glob
import os
import warnings
from copy import deepcopy
from os.path import join
import numpy as np
import obspy

from obspy.core import Stream
from obspy.core.event import Origin
from obspy.core.util.attribdict import AttribDict
from obspy.geodetics import gps2dist_azimuth
from mtuq.dataset.base import DatasetBase
from mtuq.util.signal import check_time_sampling
from mtuq.util.util import warn


class Dataset(DatasetBase):
    """ Seismic data container
 
        Adds SAC-specific metadata extraction methods
    """

    def get_origin(self, event_name=None):
        """ Extract event information from SAC metadata
        """
        sac_headers = self[0][0].stats.sac

        # location
        try:
            latitude = sac_headers.evla
            longitude = sac_headers.evlo
        except (TypeError, ValueError):
            warn("Could not determine event location from sac headers. "
                  "Setting location to nan...")
            latitude = np.nan
            longitudue = np.nan

        # depth
        try:
            depth = sac_headers.evdp
        except (TypeError, ValueError):
            warn("Could not determine event depth from sac headers. "
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
            warn("Could not determine origin time from sac headers. "
                  "Setting origin time to zero...")
            origin_time = obspy.UTCDateTime(0)

        return Origin(
            time=origin_time,
            longitude=sac_headers.evlo,
            latitude=sac_headers.evla,
            depth=depth * 1000.0,
        )


    def get_stations(self):
        """ Extract station information from SAC metadata
        """
        stations = []
        for data in self:
            station = self._copy(data[0].stats)
          
            try:
                station.channels = []
                for trace in data:
                    station.channels += [trace.stats.channel]
            except:
                raise Exception(
                    "Could not determine channel names from obspy stream.")

            try:
                station_latitude = station.sac.stla
                station_longitude = station.sac.stlo
                station.update({
                    'latitude': station_latitude,
                    'longitude': station_longitude})
            except:
                raise Exception(
                    "Could not determine station location from SAC headers.")

            try:
                station.update({
                    'station_elevation': station.sac.stel,
                    'station_depth': station.sac.stdp})
            except:
                pass

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

                station.update({
                    'catalog_latitude': event_latitude,
                    'catalog_longitude': event_longitude,
                    'catalog_distance': distance,
                    'catalog_azimuth': azimuth,
                    'catalog_backazimuth': back_azimuth})

            except:
                warn("Could not determine event location from SAC headers.")

            station.id = '.'.join((
                station.network,
                station.station,
                station.location))

            stations += [station]

        return stations


    def _copy(self, stats):
        stats = deepcopy(stats)
        stats.channel = None
        return stats


def reader(path, wildcard='*.sac', event_name=None, verbose=False):
    """ Reads SAC traces, sorts by station, and returns MTUQ Dataset

     Additional processing would be required if the time sampling varies from
     one channel to another for a given station; for now, inconsistent time
     sampling results in an exception
    """
    if not event_name:
        event_name = os.path.basename(path)

    # read traces one at a time
    data = Stream()
    for filename in glob.glob(join(path, wildcard)):
        try:
            data += obspy.read(filename, format='sac')
        except:
            if verbose:
                print('Not a SAC file: %f' % filename)

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

    # create MTUQ Dataset
    dataset = Dataset(id=event_name)
    for id, stream in data_sorted.items():
        assert check_time_sampling(stream), NotImplementedError(
            "Time sampling differs from trace to trace.")
        stream.id = id
        dataset += stream

    return dataset


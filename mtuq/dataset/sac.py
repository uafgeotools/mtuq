
import glob
import os
import warnings
from copy import deepcopy
from os.path import join
import numpy as np
import obspy

from obspy.core import Stream
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth
from mtuq.dataset.base import DatasetBase
from mtuq.util.signal import check_time_sampling
from mtuq.util.util import AttribDict, warn


class Dataset(DatasetBase):
    """ Seismic data container
 
        Adds SAC-specific metadata extraction methods
    """

    def get_origin(self, id=None, event_name=None):
        """ Extracts event metadata from SAC headers
        """
        if id:
            index = self._get_index(id)
        else:
            index = -1

        data = self.__list__[index]
        sac_headers = data[0].meta.sac


        # if hypocenter is included as an inversion parameter, then we 
        # cannot rely on any of the following metadata, which are likely based
        # on catalog locations or other preliminary information
        try:
            latitude = sac_headers.evla
            longitude = sac_headers.evlo
        except (TypeError, ValueError):
            warn("Could not determine event location from sac headers. "
                  "Setting location to nan...")
            latitude = np.nan
            longitudue = np.nan

        try:
            depth = sac_headers.evdp
        except (TypeError, ValueError):
            warn("Could not determine event depth from sac headers. "
                 "Setting depth to nan...")
            depth = 0.

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


    def get_station(self, id=None):
        """ Extracts station metadata from SAC headers
        """
        if id:
            index = self._get_index(id)
        else:
            index = -1

        data = self.__list__[index]
        sac_headers = data[0].meta.sac

        meta = AttribDict({
            'network': data[0].meta.network,
            'station': data[0].meta.station,
            'location': data[0].meta.location,
            'sac': sac_headers,
            'id': '.'.join([
                data[0].meta.network,
                data[0].meta.station,
                data[0].meta.location])})

        meta.update({
            'starttime': data[0].meta.starttime,
            'endtime': data[0].meta.endtime,
            'npts': data[0].meta.npts,
            'delta': data[0].meta.delta})

        try:
            meta.channels = []
            for trace in data:
                meta.channels += [trace.stats.channel]
        except:
            raise Exception(
                "Could not determine channel names.")

        try:
            station_latitude = sac_headers.stla
            station_longitude = sac_headers.stlo
            meta.update({
                'latitude': station_latitude,
                'longitude': station_longitude})
        except:
            raise Exception(
                "Could not determine station location from SAC headers.")

        try:
            meta.update({
                'station_elevation': sac_headers.stel,
                'station_depth': sac_headers.stdp})
        except:
            pass


        try:
            origin = self.get_origin(id)
        except:
            origin = None


        try:
            meta.update({
                'catalog_latitude': origin.latitude,
                'catalog_longitude': origin.longitude,
                'catalog_depth': origin.depth})

        except:
            print("Could not determine event location from SAC headers.")


        try:
            distance, azimuth, back_azimuth = obspy.geodetics.gps2dist_azimuth(
                station_latitude,
                station_longitude,
                origin.latitude,
                origin.longitude)

            meta.update({
                'catalog_distance': distance/1000.,
                'catalog_azimuth': azimuth,
                'catalog_backazimuth': back_azimuth})

        except:
            print("Could not determine event distance.")


        try:
            meta.update({
                'catalog_origin_time': origin.time})

        except:
            print("Could not determine origin time.")


        return meta



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
        stream.npts = data[0].meta.npts
        stream.delta = data[0].meta.delta
        stream.starttime = data[0].meta.starttime
        stream.endtime = data[0].meta.endtime

        stream.id = id
        dataset += stream

    return dataset


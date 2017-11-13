"""
given a list of sac filenames corresponding to an event, returns

- an obspy stream containing all traces associated with the event
- an obspy inventory containing all stations associated with the event
- an obspy catalog containing the event itself

"""

import glob
import os
import warnings

from os.path import join

import numpy as np
import obspy
from obspy.core import Stream, Stats, Trace
from obspy.core.event import Event, Origin
from obspy.core.inventory import Inventory, Station
from obspy.core.util.attribdict import AttribDict


def get_traces(files):
    dat = obspy.core.Stream()
    for _i, filename in enumerate(files):
        dat += obspy.read(filename, format='sac')

        try:
            # append station location to stats
            tr = dat[_i]
            sac_headers = tr.stats.sac
            tr.stats['latitude'] = sac_headers.stla
            tr.stats['longitude'] = sac_headers.stlo
        except:
            warnings.warn("Could not determine station location from sac headers.")

    return dat


def get_origin(data, event_name=None):
    sac_headers = data[0].stats.sac

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

#    return AttribDict({
#        'resource_id'=event_name,
#        'time':origin_time,
#        'longitude':longitude,
#        'latitude':latitude,
#        'depth':depth,
#        })

    return Origin(
        time=origin_time,
        longitude=sac_headers.evlo,
        latitude=sac_headers.evla,
        depth=depth * 1000.0,
    )


def get_event(event_name, origin):
    """ Creates event object based on sac metadata
    """
    ev = Event(resource_id=_get_resource_id(event_name, "event"),
        event_type="earthquake")
    #ev.event_descriptions.append(EventDescription(text=event_name,
    #    type="earthquake name"))
    #ev.comments.append(Comment(
    #    text="Hypocenter catalog: %s" % hypocenter_catalog,
    #    force_resource_id=False))
    ev.origins.append(origin)

    return ev


def get_stations(data):
    """ Creates station object based on sac metadata
    """
    stations = []
    for tr in data:
        station_name = 'dummy_name'
        station_code = 'dummy_code'
        try:
            latitude = tr.stats.sac.stla
            longitude = tr.stats.sac.stlo
            elevation = 0.
        except:
            warnings.warn("Could not determine station location from sac headers.")
        stations += [Station(
           station_code,
           latitude,
           longitude,
           elevation)]
    return stations


def _get_resource_id(res_name, res_type, tag=None):
    """
    Helper function to create consistent resource ids
    """
    res_id = "smi:local/%s/%s" % (res_name, res_type)
    if tag is not None:
        res_id += "#" + tag
    return res_id


def read(path, wildcard='*.sac'):
    """ Reads SAC files
    """
    event_name = os.path.dirname(path)
    filenames = glob.glob(join(path, wildcard))
    data = get_traces(filenames)
    return data



# debugging
if __name__=='__main__':
    read('/u1/uaf/rmodrak/packages/capuaf/20090407201255351')


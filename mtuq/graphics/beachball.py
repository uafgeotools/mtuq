
#
# graphics/beachball.py - first motion "beachball" plots
#

# To correctly plot focal mechanims, MTUQ uses Generic Mapping Tools (GMT).

# If GMT >=6.0.0 executables are not found on the system path, MTUQ falls 
# back to ObsPy. As described in the following GitHub issue, ObsPy 
# focal mechanism plots suffer from  plotting artifacts:

# https://github.com/obspy/obspy/issues/2388


import obspy.imaging.beachball
import os
import matplotlib.pyplot as pyplot
import numpy as np
import subprocess

from matplotlib import colors
from mtuq.event import MomentTensor
from mtuq.graphics._gmt import _parse_filetype, _get_format_arg
from mtuq.util import asarray, warn
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.geodetics import kilometers2degrees as _to_deg
from obspy.taup import TauPyModel
from six import string_types



def plot_beachball(filename, mt, stations, origin, **kwargs):
    """ Plots focal mechanism and station locations

    .. rubric :: Required arguments

    ``filename`` (`str`):
    Name of output image file

    ``mt`` (`mtuq.MomentTensor`):
    Moment tensor object

    ``stations`` (`list` of `Station` objects):
    Stations from which azimuths and takeoff angles are calculated

    ``origin`` (`Origin` object):
    Origin used to define center


    .. rubric :: Optional keyword arguments

    ``polarities`` (`list`)
    Observed polarities corresponding to given stations
    (not yet implemented)

    ``add_crosshair`` (`bool`)
    Marks hypocenter with crosshair

    ``add_station_labels`` (`bool`)
    Displays stations names

    ``fill_color`` (`str`)
    Used for plotting focal mechanism

    ``marker_color`` (`str`)
    Used for station markers


    """
    from mtuq.graphics._gmt import gmt_major_version

    if type(mt)!=MomentTensor:
        mt = MomentTensor(mt)

    try:
        assert gmt_major_version() >= 6
        backend = _plot_beachball_gmt

    except:
        backend = _plot_beachball_obspy

    backend(filename, mt, stations, origin, **kwargs)



def _plot_beachball_obspy(filename, mt, stations, origin, polarities=None, 
    fill_color='gray', marker_color='black', **kwargs):
    """ Plots focal mechanism using ObsPy
    """
    warn("""
        WARNING

        Generic Mapping Tools (>=6.0.0) executables not found on system path.
        Falling back to ObsPy.

        """)

    obspy.imaging.beachball.beachball(
        mt.as_vector(), size=200, linewidth=2, facecolor=fill_color)

    pyplot.savefig(filename)
    pyplot.close()



def _plot_beachball_gmt(filename, mt, stations, origin, polarities=None,
    taup_model='ak135', add_station_labels=True, 
    fill_color='gray', marker_color='black'):


    filename, filetype = _parse_filetype(filename)
    format_arg = _get_format_arg(filetype)

    # parse optional arguments
    if add_station_labels:
        label_arg = '-T+jCB'
    else:
        label_arg = ''

    if fill_color:
        rgb = 255*asarray(colors.to_rgba(fill_color)[:3])

    if marker_color:
        marker_rgb = 255*asarray(colors.to_rgba(marker_color)[:3])


    if origin and stations and polarities:
        #
        # plots beachball, station locations, and polarity fits
        #

        _write_polarities('tmp.'+filename+'.pol',
            stations, origin, polarities, taup_model)

        subprocess.call(script1 % (
            #psmeca args
            *rgb, filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #pspolar args
            'tmp.'+filename+'.sta', *marker_rgb, *marker_rgb, label_arg, filename+'.ps',

            #psconvert args
            filename+'.ps', filename, format_arg,

            ), shell=True)


    elif origin and stations:
        #
        # plots beachball and station locations
        #

        _write_stations('tmp.'+filename+'.sta',
            stations, origin, taup_model)

        subprocess.call(script2 % (
            #psmeca args
            *rgb, filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #pspolar args
            'tmp.'+filename+'.sta', *marker_rgb, *marker_rgb, label_arg, filename+'.ps',

            #psconvert args
            filename+'.ps', filename, format_arg,

            ), shell=True)

    else:
        #
        # plots beachball only
        #

        subprocess.call(script3 % (
            #psmeca args
            *rgb, filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #psconvert args
            filename+'.ps', filename, format_arg,

            ), shell=True)


#
# utility functions
#

def get_takeoff_angle(taup, source_depth_in_km, **kwargs):
    try:
        arrivals = taup.get_travel_times(source_depth_in_km, **kwargs)

        phases = []
        for arrival in arrivals:
            phases += [arrival.phase.name]

        if 'p' in phases:
            return arrivals[phases.index('p')].incident_angle

        elif 'P' in phases:
            return arrivals[phases.index('P')].incident_angle
        else:
            raise Excpetion

    except:
        # if taup fails, use dummy takeoff angle
        return None


def _write_stations(filename, stations, origin, taup_model):

    try:
        taup = TauPyModel(model=taup_model)
    except:
        taup = None

    with open(filename, 'w') as file:
        for station in stations:

            label = station.station

            distance_in_m, azimuth, _ = gps2dist_azimuth(
                origin.latitude,
                origin.longitude,
                station.latitude,
                station.longitude)

            takeoff_angle = get_takeoff_angle(
                taup, 
                origin.depth_in_m/1000.,
                distance_in_degree=_to_deg(distance_in_m/1000.),
                phase_list=['p', 'P'])

            if takeoff_angle is not None:
                file.write('%s  %f  %f\n' % (label, azimuth, takeoff_angle))


def _write_polarities(filename, stations, origin, polarities):

    raise NotImplementedError



#
# GMT SCRIPTS
#

# plots beachball, station locations, and polarity fits
# TODO - not implemented yet



# plots beachball and station locations

script2=\
'''#!/bin/bash -e

gmt psmeca -R-1.2/1.2/-1.2/1.2 -Jm0/0/5c -M -Sm9.9c -G%d/%d/%d -h1 -Xc -Yc -K << END > %s
lat lon depth   mrr   mtt   mff   mrt    mrf    mtf
0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0
END\n

# there appears to be bug? -- pspolar does not act on flags -E -G, 
# which control marker fill color

gmt pspolar %s -R-1.2/1.2/-1.2/1.2 -Jm0/0/5c -D0/0 -E%d/%d/%d -G%d/%d/%d -F -Qe -M9.9c -N -Si0.6c %s -O >> %s
gmt psconvert %s -F%s -A %s
'''


# plots beachball only

script3=\
'''#!/bin/bash -e

gmt psmeca -R-1.2/1.2/-1.2/1.2 -Jm0/0/5c -M -Sm9.9c -G%d/%d/%d -h1 -Xc -Yc << END > %s
lat lon depth   mrr   mtt   mff   mrt    mrf    mtf
0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0
END\n
gmt psconvert %s -F%s -A %s
'''



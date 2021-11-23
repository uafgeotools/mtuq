
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

from mtuq.event import MomentTensor
from mtuq.util import warn
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.geodetics import kilometers2degrees as _to_deg
from obspy.taup import TauPyModel


def plot_beachball(filename, mt, **kwargs):
    """ Plots focal mechanism of given moment tensor as PNG image

    .. rubric :: Input arguments


    ``filename`` (`str`):
    Name of output image file

    ``mt`` (`mtuq.MomentTensor`):
    Moment tensor object


    .. note::

        Ignores any given file extension: always generates a PNG file.

        (TODO: support all GMT image formats.)


    """
    from mtuq.graphics._gmt import gmt_major_version

    if type(mt)!=MomentTensor:
        mt = MomentTensor(mt)

    try:
        assert gmt_major_version() >= 6
        _plot_beachball_gmt(filename, mt)

    except:
        _plot_beachball_obspy(filename, mt)


def _plot_beachball_obspy(filename, mt):
    """ Plots focal mechanism using ObsPy
    """
    warn("""
        WARNING

        Generic Mapping Tools (>=6.0.0) executables not found on system path.
        Falling back to ObsPy.

        """)

    obspy.imaging.beachball.beachball(
        mt.as_vector(), size=200, linewidth=2, facecolor=gray)

    pyplot.savefig(filename)
    pyplot.close()



def plot_beachball(filename, mt, origin=None, stations=None, polarities=None,
    model='ak135', write_station_labels=False, fill_color=None):


    # parse filename
    if filename.endswith('.png'):
        filename = filename[:-4]

    if filename.endswith('.ps'):
        filename = filename[:-3]


    # parse optional arguments
    if write_station_labels:
        station_label_arg = '-T+jCB'
    else:
        station_label_arg = ''

    if fill_color:
        raise NotImplementedError



    if origin and stations and polarities:

        _write_polarities('tmp.'+filename+'.pol',
            stations, origin, polarities, model)

        subprocess.call(script1 % (
            #psmeca args
            filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #pspolar args
            'tmp.'+filename+'.pol', station_label_arg, filename+'.ps',

            #psconvert args
            filename+'.ps',

            ), shell=True)


    elif origin and stations:

        _write_stations('tmp.'+filename+'.sta',
            stations, origin, model)

        subprocess.call(script2 % (
            #psmeca args
            filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #pspolar args
            'tmp.'+filename+'.sta', station_label_arg, filename+'.ps',

            #psconvert args
            filename+'.ps',

            ), shell=True)

    else:
        subprocess.call(script3 % (
            #psmeca args
            filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #psconvert args
            filename+'.ps',

            ), shell=True)


#
# utility functions
#

def get_takeoff_angle(taup_model, source_depth_in_km, **kwargs):
    try:
        arrivals = taup_model.get_travel_times(source_depth_in_km, **kwargs)

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


def _write_stations(filename, stations, origin, model):

    try:
        taup_model = TauPyModel(model=model)
    except:
        taup_model = None

    with open(filename, 'w') as file:
        for station in stations:

            label = station.station

            distance_in_m, azimuth, _ = gps2dist_azimuth(
                origin.latitude,
                origin.longitude,
                station.latitude,
                station.longitude)

            takeoff_angle = get_takeoff_angle(
                taup_model, 
                origin.depth_in_m/1000.,
                distance_in_degree=_to_deg(distance_in_m/1000.),
                phase_list=['p', 'P'])

            if takeoff_angle is not None:
                file.write('%s  %f  %f\n' % (label, azimuth, takeoff_angle))


def _write_polarities(filename, stations, origin, polarities):

    raise NotImplementedError


gray = [0.667, 0.667, 0.667]



#
# GMT SCRIPTS
#

# SCRIPT1 - plots beachball, station locations, and polarity fits

# TODO - not implemented yet



# SCRIPT2 - plots beachball and station locations

script2=\
'''#!/bin/bash -e

gmt psmeca -R-1.2/1.2/-1.2/1.2 -Jm0/0/5c -M -Sm9.9c -Ggrey50 -h1 -Xc -Yc -K << END > %s
lat lon depth   mrr   mtt   mff   mrt    mrf    mtf
0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0
END\n
gmt pspolar %s -R -J -D0/0 -F -M9.9c -N -Si0.4c %s -O >> %s
gmt psconvert %s -A -Tg
'''


# SCRIPT3 - plots beachball only

script3=\
'''#!/bin/bash -e

gmt psmeca -R-1.2/1.2/-1.2/1.2 -Jm0/0/5c -M -Sm9.9c -Ggrey50 -h1 -Xc -Yc << END > %s
lat lon depth   mrr   mtt   mff   mrt    mrf    mtf
0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0
END\n
gmt psconvert %s -A -Tg
'''



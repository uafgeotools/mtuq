
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

from glob import glob
from mtuq.event import MomentTensor
from mtuq.graphics._gmt import _parse_filetype, _get_format_arg, _safename,\
    exists_gmt, gmt_major_version
from mtuq.graphics._pygmt import exists_pygmt
from mtuq.util import asarray, to_rgb, warn
from mtuq.util.polarity import calculate_takeoff_angle
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

    ``origin`` (`mtuq.Origin`):
    Origin object


    .. rubric :: Optional arguments

    ``crosshair`` (`bool`)
    Marks hypocenter with crosshair

    ``add_station_labels`` (`bool`)
    Displays station names

    ``add_station_markers`` (`bool`)
    Displays station markers

    ``fill_color`` (`str`)
    Used for plotting beachball

    ``marker_color`` (`str`)
    Used for station markers


    """

    if type(mt)!=MomentTensor:
        raise TypeError

    try:
        assert exists_gmt()
        assert gmt_major_version() >= 6
        _plot_beachball_gmt(filename, mt, stations, origin, **kwargs)
        return
    except:
        pass

    try:
        warn("plot_beachball: Falling back to ObsPy.")
        from matplotlib import pyplot
        obspy.imaging.beachball.beachball(
            mt.as_vector(), size=200, linewidth=2, facecolor=fill_color)
        pyplot.savefig(filename)
        pyplot.close()
    except:
        warn("plot_beachball failed. No figure generated.")



def plot_polarities(filename, polarities, mt, stations, origin, **kwargs):
    """ Plots first-motion polarities

    .. rubric :: Required arguments

    ``filename`` (`str`):
    Name of output image file

    ``stations`` (`list` of `Station` objects):
    Stations from which azimuths and takeoff angles are calculated

    ``origin`` (`mtuq.Origin`):
    Origin object

    ``polarities`` (`dict`)
    Polarities dictionary

    ``mt`` (`mtuq.MomentTensor`):
    Moment tensor object

    """
    raise NotImplementedError


#
# GMT implementation
#

GMT_REGION = '-R-1.2/1.2/-1.2/1.2'
GMT_PROJECTION = '-Jm0/0/5c'


def _plot_beachball_gmt(filename, mt, stations, origin,
    taup_model='ak135', add_station_markers=True, add_station_labels=True,
    fill_color='gray', marker_color='black'):


    filename, filetype = _parse_filetype(filename)
    format_arg = _get_format_arg(filetype)

    # parse optional arguments
    label_arg = ''
    if add_station_labels:
        label_arg = '-T+jCB'

    if fill_color:
        rgb = to_rgb(fill_color)

    if marker_color:
        rgb2 = to_rgb(marker_color)


    if stations and origin and (add_station_markers or add_station_labels):
        #
        # plots beachball and stations
        #
        ascii_table = _safename('tmp.'+filename+'.sta')
        _write_stations(ascii_table, stations, origin, taup_model)

        subprocess.call(
            '#!/bin/bash -e\n'

            'gmt psmeca %s %s -M -Sm9.9c -G%d/%d/%d -h1 -Xc -Yc -K << END > %s\n'
            'lat lon depth   mrr   mtt   mpp   mrt    mrp    mtp\n'
            '0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0\n'
            'END\n\n'

            'gmt pspolar %s %s %s -D0/0 -E%d/%d/%d -G%d/%d/%d -F -Qe -M9.9c -N -Si0.6c %s -O >> %s\n'

            'gmt psconvert %s -F%s -A %s\n'
             %
        (
            #psmeca args
            GMT_REGION, GMT_PROJECTION, *rgb, filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #pspolar args
            GMT_REGION, GMT_PROJECTION, ascii_table, *rgb2, *rgb2, label_arg, filename+'.ps',

            #psconvert args
            filename+'.ps', filename, format_arg,

        ), shell=True)

    else:
        #
        # plots beachball only
        #
        subprocess.call(
            '#!/bin/bash -e\n'

            'gmt psmeca %s %s -M -Sm9.9c -G%d/%d/%d -h1 -Xc -Yc << END > %s\n'
            'lat lon depth   mrr   mtt   mpp   mrt    mrp    mtp\n'
            '0.  0.  10.    %e     %e    %e    %e     %e     %e 25 0 0\n'
            'END\n\n'

            'gmt psconvert %s -F%s -A %s\n'
            %
        (
            #psmeca args
            GMT_REGION, GMT_PROJECTION, *rgb, filename+'.ps',

            #psmeca table
            *mt.as_vector(),

            #psconvert args
            filename+'.ps', filename, format_arg,

        ), shell=True)


    # remove temporary files
    for filename in glob(_safename('tmp.'+filename+'*')):
        os.remove(filename)


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

            takeoff_angle = calculate_takeoff_angle(
                taup,
                origin.depth_in_m/1000.,
                distance_in_degree=_to_deg(distance_in_m/1000.),
                phase_list=['p', 'P'])

            if takeoff_angle is not None:
                file.write('%s  %f  %f\n' % (label, azimuth, takeoff_angle))

#
# PyGMT implementation (experimental)
#

def _plot_beachball_pygmt(filename, mt, stations, origin,
    taup_model='ak135', add_station_labels=True, add_station_markers=True,
    fill_color='gray', marker_color='black'):

    import pygmt
    from pygmt.helpers import build_arg_string, use_alias

    gmt_region    = [-1.2, 1.2, -1.2, 1.2]
    gmt_projection= 'm0/0/5c'
    gmt_scale     = '9.9c'

    fig = pygmt.Figure()

    #
    # plot the beachball itself
    #
    fig.meca(
        # basemap arguments
        region=gmt_region,
        projection=gmt_projection,
        scale=gmt_scale,

        # lon, lat, depth, mrr, mtt, mpp, mrt, mrp, mtp, exp, lon2, lat2
        spec=(0, 0, 10, *mt.as_vector(), 25, 0, 0),
        convention="mt",

        # face color
        G='grey50',

        N=False,
        M=True,
        )

    #
    # add station markers and labels
    #
    if stations and origin:
        _write_stations(_safename('tmp.'+filename+'.sta'),
            stations, origin, taup_model)

        @use_alias(
            D='offset',
            J='projection',
            M='scale',
            S='symbol',
            E='ext_fill',
            G='comp_fill',
            F='background',
            Qe='ext_outline',
            Qg='comp_outline',
            Qf='mt_outline',
            T='station_labels'
        )
        def _pygmt_polar(ascii_table, **kwargs):
            arg_string = " ".join([ascii_table, build_arg_string(kwargs)])
            with pygmt.clib.Session() as session:
                session.call_module('polar',arg_string)

        _pygmt_polar(
            _safename('tmp.'+filename+'.sta'),

            # basemap arguments
            projection=gmt_projection,
            scale=gmt_scale,

            station_labels='+f0.18c',
            offset='0/0',
            symbol='t0.40c',
            comp_fill='grey50',
            )


        fig.savefig(filename)


def _plot_polarities_pygmt():
    # calculate theoretical polarities
    calculate_polarities()

    # case 1 - observed and synthetic both positive
    _pygmt_polar()

    # case 2 - observed and synthetic both negative
    _pygmt_polar()

    # case 3 - observed positive, synthetic negative
    _pygmt_polar()

    # case 4 - observed negative, synthetic positive
    _pygmt_polar()

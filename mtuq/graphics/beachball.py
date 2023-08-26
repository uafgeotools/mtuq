#
# graphics/beachball.py - first motion "beachball" plots
#

# - uses PyGMT if present to plot beachballs
# - if PyGMT is not present, attempts to fall back to GMT >=6
# - if GMT >=6 is not present, attempts to fall back to ObsPy


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
from mtuq.misfit.polarity import _takeoff_angle_taup
from obspy.geodetics import gps2dist_azimuth
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

    ``add_station_labels`` (`bool`):
    Displays station names

    ``add_station_markers`` (`bool`):
    Displays station markers

    ``fill_color`` (`str`):
    Color used for beachball

    ``marker_color`` (`str`):
    Color used for station markers

    ``taup_model`` (`str`):
    Name of built-in ObsPy TauP model or path to custom ObsPy TauP model,
    used for takeoff angle calculations

    """

    if type(mt)!=MomentTensor:
        raise TypeError

    if exists_pygmt():
        _plot_beachball_pygmt(filename, mt, stations, origin, **kwargs)
        return

    if exists_gmt() and gmt_major_version() >= 6:
        _plot_beachball_gmt(filename, mt, stations, origin, **kwargs)
        return

    try:
          warn("plot_beachball: Falling back to ObsPy")
          from matplotlib import pyplot
          obspy.imaging.beachball.beachball(
              mt.as_vector(), size=200, linewidth=2, facecolor=fill_color)
          pyplot.savefig(filename)
          pyplot.close()
    except:
        warn("plot_beachball: Plotting failed")


def plot_polarities(filename, observed, predicted, stations, origin, mt, **kwargs):
    """ Plots first-motion polarities

    .. rubric :: Required arguments

    ``filename`` (`str`):
    Name of output image file

    ``observed`` (`list` or `dict`)
    Observed polarities for all stations (+1 positive, -1 negative, 0 unpicked)

    ``predicted`` (`list` or `dict`)
    Predicted polarities for all stations (+1 positive, -1 negative)

    ``stations`` (`list`):
    List containting station names, azimuths and takeoff angles

    ``origin`` (`mtuq.Origin`):
    Origin object

    ``mt`` (`mtuq.MomentTensor`):
    Moment tensor object

    """
    if exists_pygmt():
        _plot_polarities_pygmt(filename, observed, predicted,
            stations, origin, mt, **kwargs)

    else:
        raise Exception('Requires PyGMT')


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
    for _filename in glob(_safename('tmp.'+filename+'*')):
        os.remove(_filename)


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

            takeoff_angle = _takeoff_angle_taup(
                taup,
                origin.depth_in_m/1000.,
                _to_deg(distance_in_m/1000.))

            if takeoff_angle is not None:
                file.write('%s  %f  %f\n' % (label, azimuth, takeoff_angle))


#
# PyGMT implementation (experimental)
#

PYGMT_REGION    = [-1.2, 1.2, -1.2, 1.2]
PYGMT_PROJECTION= 'm0/0/5c'
PYGMT_SCALE     = '9.9c'


def _plot_beachball_pygmt(filename, mt, stations, origin,
    taup_model='ak135', add_station_labels=True, add_station_markers=True,
    fill_color='gray', marker_color='black'):

    import pygmt
    fig = pygmt.Figure()

    #
    # plot the beachball itself
    #
    _meca_pygmt(fig, mt)

    #
    # add station markers and labels
    #
    if stations and origin:
        _write_stations(_safename('tmp.'+filename+'.sta'),
            stations, origin, taup_model)

        _polar1(
            _safename('tmp.'+filename+'.sta'),

            # basemap arguments
            projection=PYGMT_PROJECTION,
            scale=PYGMT_SCALE,

            station_labels='+jCB',
            offset='0/0',
            symbol='i0.60c',
            comp_fill='black',
            ext_fill='black',
            background=True,
            )

    fig.savefig(filename)

    # remove temporary files
    for _filename in glob(_safename('tmp.'+filename+'*')):
        os.remove(_filename)


def _plot_polarities_pygmt(filename, observed, predicted, 
    stations, origin, mt, **kwargs):

    import pygmt

    if observed.size != predicted.size:
        raise Exception('Inconsistent dimensions')

    if observed.size != len(stations):
        raise Exception('Inconsistent dimensions')

    observed = observed.flatten()
    predicted = predicted.flatten()

    up_matched = [station for _i, station in enumerate(stations)
        if observed[_i] == predicted[_i] == 1]

    down_matched = [station for _i, station in enumerate(stations)
        if observed[_i] == predicted[_i] == -1]

    up_unmatched = [station for _i, station in enumerate(stations)
        if (observed[_i] == 1) and (predicted[_i] == -1)]

    down_unmatched = [station for _i, station in enumerate(stations)
        if (observed[_i] == -1) and (predicted[_i] == 1)]

    unpicked = [station for _i, station in enumerate(stations)
        if (observed[_i] != +1) and (observed[_i] != -1)]


    fig = pygmt.Figure()

    # the beachball itself
    _meca_pygmt(fig, mt)

    # observed and synthetic both positive
    _polar2(up_matched, symbol='t0.40c', comp_fill='green')

    # observed and synthetic both negative
    _polar2(down_matched, symbol='i0.40c', ext_fill='green')

    # observed positive, synthetic negative
    _polar2(up_unmatched, symbol='t0.40c', comp_fill='red')

    # observed negative, synthetic positive
    _polar2(down_unmatched, symbol='i0.40c', ext_fill='red')

    fig.savefig(filename)


def _meca_pygmt(fig, mt):
    fig.meca(
        # lon, lat, depth, mrr, mtt, mpp, mrt, mrp, mtp, lon2, lat2
        np.array([0, 0, 10, *mt.as_vector(), 0, 0]),

        scale=PYGMT_SCALE,
        convention="mt",

        # basemap arguments
        region=PYGMT_REGION,
        projection=PYGMT_PROJECTION,

        compressionfill='grey50',
        no_clip=False,
        M=True,
        )


# ugly workarounds like those below necessary until PyGMT itself implements 
# GMT polar

def _polar1(ascii_table, **kwargs):
    import pygmt
    from pygmt.helpers import build_arg_string, use_alias

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
    def __polar1(ascii_table, **kwargs):

        arg_string = " ".join([ascii_table, build_arg_string(kwargs)])
        with pygmt.clib.Session() as session:
            session.call_module('polar',arg_string)

    __polar1(ascii_table, **kwargs)


def _polar2(stations, **kwargs):
    import pygmt
    from pygmt.helpers import build_arg_string, use_alias

    @use_alias(
        D='offset',
        J='projection',
        M='size',
        S='symbol',
        E='ext_fill',
        G='comp_fill',
        F='background',
        Qe='ext_outline',
        Qg='comp_outline',
        Qf='mt_outline',
        T='station_labels'
    )
    def __polar2(stations, **kwargs):

        # apply defaults
        kwargs = {
            'D' : '0/0',
            'J' : 'm0/0/5c',
            'M' : '9.9c',
            'T' : '+f0.18c',
            'R': '-1.2/1.2/-1.2/1.2',
            **kwargs,
            }

        with open('_tmp_polar2', 'w') as f:
            for station in stations:

                label = station.network+'.'+station.station
                try:
                    if station.polarity > 0:
                        polarity = '+'
                    elif station.polarity < 0:
                        polarity = '-'
                    else:
                        polarity = '0'
                except:
                    polarity = '0'

                f.write("%s %s %s %s\n" % (
                    label, station.azimuth, station.takeoff_angle, polarity))

        arg_string = " ".join(['_tmp_polar2', build_arg_string(kwargs)])
        with pygmt.clib.Session() as session:
            session.call_module('polar',arg_string)
        os.remove('_tmp_polar2')


    __polar2(stations, **kwargs)



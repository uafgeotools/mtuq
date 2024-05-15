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
from mtuq.graphics.uq._matplotlib import _hammer_projection
from mtuq.util import asarray, to_rgb, warn
from mtuq.misfit.polarity import _takeoff_angle_taup
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees as _to_deg
from obspy.taup import TauPyModel
from six import string_types
from mtuq.util.beachball import offset_fibonacci_sphere,\
    convert_sphere_points_to_angles, lambert_azimuthal_equal_area_projection,\
    estimate_angle_on_lune, rotate_tensor, polarities_mt, rotate_points, _project_on_sphere


def plot_beachball(filename, mt, stations, origin, backend=None, **kwargs):
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

    if backend is None:
        backend = _plot_beachball_matplotlib
        backend(filename, mt, stations, origin, **kwargs)
        return
    elif backend == _plot_beachball_pygmt and exists_pygmt():
        backend(filename, mt, stations, origin, **kwargs)
        return
    elif backend == _plot_beachball_gmt and exists_gmt() and gmt_major_version() >= 6:
        backend(filename, mt, stations, origin, **kwargs)
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


def _plot_beachball_matplotlib(filename, mt_arrays, stations=None, origin=None, lat_lons=None, 
                               scale=None, fig=None, ax=None, taup_model='ak135', color='gray', 
                               lune_rotation=False, **kwargs):
    
    from scipy.interpolate import griddata

    if lat_lons is not None:
        if len(lat_lons) != len(mt_arrays):
            raise ValueError("This function either takes a single moment tensor or a list of moment\
                              tensors with corresponding latitudes and longitudes. lat_lons must be\
                              provided and have the same length as mt_arrays")
    else:
        if isinstance(mt_arrays, MomentTensor):
            lat_lons = np.array([[0, 0]])
            mt_arrays = mt_arrays.as_vector().reshape(1, 6)
        elif np.shape(mt_arrays) == (6,):
            lat_lons = np.array([[0, 0]])
            mt_arrays = mt_arrays.reshape(1, 6)
        elif np.shape(mt_arrays) == (1, 6):
            lat_lons = np.array([[0, 0]])
        else:
            raise ValueError("This function either takes a single moment tensor or a list of moment\
                              tensors with corresponding latitudes and longitudes. You're trying to\
                              provide a single object that is not a valid moment tensor.")

    if scale is None:
        scale = 2.  # Default scale if not provided

    # Check if axes are provided
    if fig is None or ax is None:
        fig, ax = pyplot.subplots(figsize=(1171/300, 1171/300), dpi=300)
        mode = 'MT_Only'
    else:
        mode = 'Scatter MT'

    # Generate points on the sphere using the Fibonacci method (common for all tensors)
    if mode == 'MT_Only':
        points = offset_fibonacci_sphere(50000, 0, 360)
    elif mode == 'Scatter MT':
        points = offset_fibonacci_sphere(5000, 0, 360)
    upper_hemisphere_mask = points[:, 1] >= 0
    takeoff_angles, azimuths = convert_sphere_points_to_angles(points[upper_hemisphere_mask])
    lambert_points = lambert_azimuthal_equal_area_projection(points[upper_hemisphere_mask], hemisphere='upper')
    x_proj, z_proj = lambert_points.T

    # Creating a meshgrid for interpolation (common for all tensors)
    if mode == 'MT_Only':
        xi, zi = np.linspace(x_proj.min(), x_proj.max(), 600), np.linspace(z_proj.min(), z_proj.max(), 600)
    elif mode == 'Scatter MT':
        xi, zi = np.linspace(x_proj.min(), x_proj.max(), 200), np.linspace(z_proj.min(), z_proj.max(), 200)
    xi, zi = np.meshgrid(xi, zi)
    
    for mt_array, lat_lon in zip(mt_arrays, lat_lons):

        if isinstance(mt_array, MomentTensor):
            mt_array = mt_array.as_vector().reshape(1, 6)
        elif np.shape(mt_array) != (6):
            mt_array = mt_array.reshape(1, 6)
        elif np.shape(mt_array) != (1, 6):
            raise ValueError("Each moment tensor array must be of shape (1, 6)")

        # Position and rotation on the lune
        if lune_rotation:
            lat, lon = _hammer_projection(*lat_lon)
            angle = estimate_angle_on_lune(lon, lat)
        else:
            lat, lon = lat_lon
            angle = 0

        XI, ZI = rotate_points(xi.copy(), zi.copy(), angle)  # Rotate grid to match the direction of the pole

        # Polarities and radiation pattern calculation
        polarities, radiations = polarities_mt(rotate_tensor(mt_array), takeoff_angles, azimuths)
        radiations_grid = griddata((x_proj, z_proj), radiations, (XI, ZI), method='cubic')  # Project according to the rotation

        # Plotting the radiation pattern
        ax.contourf(lat + xi * scale, lon + zi * scale, radiations_grid, [-np.inf, 0], colors=['white'], alpha=1, zorder=100, antialiased=True)
        ax.contourf(lat + xi * scale, lon + zi * scale, radiations_grid, [0, np.inf], colors=[color], alpha=1, zorder=100, antialiased=True)
        outer_circle = pyplot.Circle((lat, lon), scale-0.001*scale, color='gray', fill=False, linewidth=0.5, zorder=100)
        ax.add_artist(outer_circle)

    # Adjusting the axes properties
    ax.set_aspect('equal')
    if mode == 'MT_Only':
        ax.axis('off')
        ax.set_xlim(-1.1*scale, 1.1*scale)
        ax.set_ylim(-1.1*scale, 1.1*scale)

    if stations and origin:
        _write_stations_matplotlib(stations, origin, taup_model, ax, scale=scale)

    if filename:
        pyplot.tight_layout(pad=-0.8)
        fig.savefig(filename, dpi=300)
        pyplot.close(fig)
    return


def _write_stations_matplotlib(stations, origin, taup_model, ax, scale=1):

    try:
        taup = TauPyModel(model=taup_model)
    except:
        taup = None

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

        x,y,z = _project_on_sphere(takeoff_angle, azimuth,scale=1)
        # Project with lambert azimuthal equal area projection
        if takeoff_angle <= 90:
            projected_points = lambert_azimuthal_equal_area_projection(np.array([[x,y,z]]), hemisphere='upper')[0]*scale
            # Plot a black circle at the projected point
            ax.scatter(*projected_points, color='black', marker='o', s=15, zorder=101)
        else:
            projected_points = -lambert_azimuthal_equal_area_projection(np.array([[x,y,z]]), hemisphere='lower')[0]*scale
            # Plot a x at the projected point
            ax.scatter(*projected_points, color='black', marker='x', s=15, zorder=101)

        # Add station label with a slight offset from the projected point
        # y offset is scale*0.05
        # Make the text centered on the projected point x axis
        ax.text(projected_points[0], projected_points[1]+scale*0.05, label, ha='center', va='center', fontsize=6, zorder=101)
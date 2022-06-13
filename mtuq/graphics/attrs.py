
import matplotlib
import os
import numpy as np

from matplotlib import pyplot
from os.path import join

from mtuq.util import defaults, warn


def plot_time_shifts(dirname, attrs, stations, origin, **kwargs):
    """ Plots how time shifts vary by location and component

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'Time shift (s)',
        })

    _plot_attrs(dirname, stations, origin, attrs, 'time_shift', **kwargs)


def plot_amplitude_ratios(dirname, attrs, stations, origin, **kwargs):
    """ Plots how Aobs/Asyn varies by location and component

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_


    """
    defaults(kwargs, {
        'colormap': 'Reds',
        'label': '$A_{obs}/A_{syn}$',
        'zero_centered': False,
        })

    _plot_attrs(dirname, stations, origin, attrs, 'amplitude_ratio', **kwargs)


def plot_log_amplitude_ratios(dirname, attrs, stations, origin, **kwargs):
    """ Plots how ln(Aobs/Asyn) varies by location and component

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects):
    Used to plot station locations

    ``origin`` (`mtuq.Origin` object):
    Used to plot origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'ln($A_{obs}/A_{syn}$)',
        })

    _plot_attrs(dirname, stations, origin, attrs, 'log_amplitude_ratio', **kwargs)


def _plot_attrs(dirname, stations, origin, attrs, key,
     components=['Z', 'R', 'T'], format='png', backend=None,
     **kwargs):

    """ Reads the attribute given by `key` from the `attrs` data structure, and
    plots how this attribute varies

    Within the specified directory, a separate figure will be created for each
    component, e.g. `Z.png`, `R.png`, `T.png`.


    .. rubric ::  Keyword arguments

    ``components`` (`list`):
    Generate figures for the given components

    ``format`` (`str`):
    Image file format (defaults to `png`)

    ``backend`` (`function`):
    Backend function


    .. rubric :: Backend function

    To customize figure appearance, users can pass their own backend function.
    See `online documentation 
    <https://uafgeotools.github.io/mtuq/user_guide/06/customizing_figures.html>`_
    for details. Otherwise, defaults to a generic matplotlib `backend
    <mtuq.graphics.attrs._default_backend.html>`_.


    """

    if backend is None:
        backend = _default_backend

    if not callable(backend):
        raise TypeError


    os.makedirs(dirname, exist_ok=True)

    for component in components:
        values = []
        station_list = []

        for _i, station in enumerate(stations):
            if component not in attrs[_i]:
                continue

            values += [attrs[_i][component][key]]
            station_list += [stations[_i]]

        if len(values) > 0:
            filename = join(dirname, component+'.'+format)
            backend(filename, values, station_list, origin, **kwargs)


#
# low-level function for plotting trace attributes
#

def _default_backend(filename, values, stations, origin,
    colormap='coolwarm', zero_centered=True, colorbar=True,
    label='', width=5., height=5.):

    """ Default backend for all other `mtuq.graphics.attrs` functions

    The frontend functions perform only data manipulation. All graphics library
    calls occur in the backend

    By isolating the graphics function calls in this way, users can completely
    interchange graphics libraries (matplotlib, GMT, PyGMT, and so on)

    .. rubric::  Keyword arguments

    ``colormap`` (`str`):
    Matplotlib color palette

    ``zero_centered`` (`bool`):
    Whether or not the colormap is centered on zero

    ``colorbar`` (`bool`):
    Whether or not to display a colorbar

    ``label`` (`str`):
    Optional colorbar label


    """

    fig = pyplot.figure(figsize=(width, height))


    # generate colormap
    cmap = matplotlib.cm.get_cmap(colormap)

    if zero_centered:
        min_val = -np.max(np.abs(values))
        max_val = +np.max(np.abs(values))
    else:
        min_val = np.min(values)
        max_val = np.max(values)
 
    # plot stations
    im = pyplot.scatter(
        [station.longitude for station in stations],
        [station.latitude for station in stations], 
        s=80.,
        c=values, 
        cmap=cmap, 
        vmin=min_val,
        vmax=max_val,
        marker='^',
        )

    # plot line segments
    for _i, station in enumerate(stations):

        scaled = (values[_i]-min_val)/(max_val-min_val)
        rgb = cmap(scaled)

        pyplot.plot(
            [origin.longitude, station.longitude],
            [origin.latitude, station.latitude],
            marker=None,
            color=rgb,
            linestyle='-',
            linewidth=0.5,
            )

    # plot origin
    pyplot.plot(
        origin.longitude,
        origin.latitude,
        marker='*',
        markersize=15.,
        color='black',
        )

    # adjust ticks
    pyplot.gca().tick_params(top=True, right=True,
        labeltop=True, labelright=True)

    pyplot.locator_params(nbins=3)

    # add colorbar
    if not label:
        label = ''

    fig.colorbar(im, orientation="horizontal", pad=0.2,
        label=label)

    pyplot.savefig(filename)
    pyplot.close()



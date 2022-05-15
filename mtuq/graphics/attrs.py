
import matplotlib
import os
import numpy as np

from matplotlib import pyplot
from os.path import join

from mtuq.util import defaults, warn


def plot_time_shifts(dirname, attrs, stations, origin, **kwargs):
    """ Plots how time shifts vary geographically

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects)
    Used to extract station locations

    ``origin`` (`mtuq.Origin` object)
    Used to extract origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'Time shift (s)',
        })

    _plot_attrs(dirname, stations, origin, attrs, 'time_shift', **kwargs)


def plot_amplitude_ratios(dirname, attrs, stations, origin, **kwargs):
    """ Plots how Aobs/Asyn varies geographically

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects)
    Used to extract station locations

    ``origin`` (`mtuq.Origin` object)
    Used to extract origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_


    """
    defaults(kwargs, {
        'label': '$A_{obs}/A_{syn}$',
        'colormap': 'Reds',
        'centered': False,
        })

    _plot_attrs(dirname, stations, origin, attrs, 'amplitude_ratio', **kwargs)


def plot_log_amplitude_ratios(dirname, attrs, stations, origin, **kwargs):
    """ Plots how ln(Aobs/Asyn) varies geographically

    .. rubric :: Required input arguments

    ``dirname`` (`str`):
    Directory in which figures will be written

    ``attrs`` (`list` of `AttribDict`):
    List returned by misfit function's `collect_attributes` method

    ``stations`` (`list` of `mtuq.Station` objects)
    Used to extract station locations

    ``origin`` (`mtuq.Origin` object)
    Used to extract origin location


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_attrs.html>`_

    """
    defaults(kwargs, {
        'label': 'ln($A_{obs}/A_{syn}$)',
        })

    _plot_attrs(dirname, stations, origin, attrs, 'log_amplitude_ratio', **kwargs)


def _plot_attrs(dirname, stations, origin, attrs, key,
     components=['Z', 'R', 'T'], format='png', backend='matplotlib',
     **kwargs):

    """ Reads the attribute given by `key` from the `attrs` data structure, and
    plots how this attribute varies geographically

    Within the specified directory, a separate figure will be created for each
    component, e.g. `Z.png`, `R.png`, `T.png`.


    .. rubric :: Optional input arguments


    """

    if backend=='matplotlib':
        backend = _matplotlib

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

def _matplotlib(filename, values, stations, origin,
    add_station_labels=True, colormap='coolwarm', centered=True, label='',
    figsize=(5., 6.)):

    fig = pyplot.figure(figsize=figsize)


    # generate colormap
    cmap = matplotlib.cm.get_cmap(colormap)
    if centered:
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
    fig.colorbar(im, orientation="horizontal", pad=0.2,
        label=label)

    pyplot.savefig(filename)
    pyplot.close()


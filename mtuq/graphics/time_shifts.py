import os

from matplotlib import pyplot
from os.path import join

from mtuq.graphics.waveforms import _set_components, _prepare_synthetics
from mtuq.util import warn


def plot_time_shifts(filename, data, greens, component, misfit, stations, origin, source,
    backend='matplotlib'):

    """ For a given component, creates a "spider plot" showing how 
    time shifts vary geographically
    """
    if backend.lower()=='gmt':
        raise NotImplementedError

    # prepare synthetics
    greens = greens.select(origin)
    _set_components(data, greens)
    synthetics, _ = _prepare_synthetics(data, greens, misfit, source)

    # collect time shifts
    time_shifts, indices = _collect_time_shifts(synthetics, component)

    if len(indices)==0:
        warn("Component not present in dataset")
        return

    _save_figure(filename,
        time_shifts, [stations[_i] for _i in indices], origin, source)


def plot_time_shifts_ZRT(dirname, data, greens, misfit, stations, origin, source,
     format='png', backend='matplotlib'):

    """ For components `'Z','R','T'`, creates "spider plots" showing how 
    time shifts vary geographically

    Within the specified directory, a separate PNG figure will be created for 
    each component. Any components not present in the data will be skipped.

    """
    if backend.lower()=='gmt':
        raise NotImplementedError

    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    # prepare synthetics
    greens = greens.select(origin)
    _set_components(data, greens)
    synthetics, _ = _prepare_synthetics(data, greens, misfit, source)

    for component in ('Z','R','T'):
        filename = join(dirname, component+'.'+format)

        # collect time shifts
        time_shifts, indices = _collect_time_shifts(synthetics, component)

        if len(indices)==0:
            continue

        _save_figure(filename,
            time_shifts, [stations[_i] for _i in indices], origin, source)


def _collect_time_shifts(synthetics, component):
    """ Collects time shifts by interating over data structure
    """
    time_shifts = []
    indices = []

    for _i in range(len(synthetics)):
        stream = synthetics[_i].select(component=component)

        if len(stream)==0:
            time_shifts += []
            continue

        elif len(stream)==1:
            trace = stream[0]

        else:
            raise Exception(
                "Too many %s component traces at station %s" % 
                (component, stream[_i].station.id))

        if not hasattr(trace, 'time_shift'):
            raise Exception(
                 "Missing time_shift attribute")

        time_shifts += [trace.time_shift]
        indices += [_i]

    return time_shifts, indices


def _save_figure(filename, time_shifts, stations, origin, source, 
    cmap='seismic', station_marker_size=80, source_marker_size=15):

    """ Creates the actual "spider plot"
    """
    #
    # TODO 
    #   - replace generic source marker with beachball
    #   - implement alternative GMT version
    #

    pyplot.figure()

    # plot "spider lines"
    for station in stations:
        pyplot.plot(
            [origin.longitude, station.longitude],
            [origin.latitude, station.latitude],
            marker=None,
            color='black',
            linestyle='-',
            linewidth=0.5,
            )

    # plot stations
    pyplot.scatter(
        [station.longitude for station in stations],
        [station.latitude for station in stations], 
        s=station_marker_size,
        c=time_shifts, 
        cmap=cmap, 
        marker='^',
        )

    # plot origin
    pyplot.plot(
        origin.longitude, 
        origin.latitude,
        marker='o', 
        markersize=source_marker_size,
        )

    pyplot.savefig(filename)
    pyplot.close()


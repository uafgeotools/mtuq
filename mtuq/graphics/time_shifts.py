
import os

from matplotlib import pyplot
from os.path import join

from mtuq.graphics.waveforms import _set_components, _prepare_synthetics


MARKERSIZE=14


def plot_time_shifts(dirname, data, greens, misfit, stations, origin, source, components=['Z','R','T'], format='png', backend='matplotlib'):

    """ Creates "spider plots" showing how time shifts vary geographically
    """

    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

    # prepare synthetics
    greens = greens.select(origin)
    _set_components(data, greens)
    synthetics, _ = _prepare_synthetics(data, greens, misfit, source)

    for component in components:

        # collect time shifts
        time_shifts, indices = _collect_time_shifts(synthetics, component)

        if len(indices)==0:
            continue

        if backend.lower()=='gmt':
            raise NotImplementedError

        else:
            # generate figure
            pyplot.figure()
            _display_source(source, origin)
            _display_time_shifts(stations, origin, time_shifts, indices)
            filename = join(dirname, component+'.'+format)
            pyplot.savefig(filename)
            pyplot.close()



def _collect_time_shifts(synthetics, component):
    # iterates over Dataset to collect time shifts
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


def _display_source(source, origin):
    pyplot.plot(origin.longitude, origin.latitude,
        marker='o', markersize=MARKERSIZE)


def _display_time_shifts(stations, origin, vals, indices):
    lat = [stations[_i].latitude for _i in indices]
    lon = [stations[_i].longitude for _i in indices]

    pyplot.scatter(lon, lat, c=vals, cmap='seismic', 
        marker='^', s=MARKERSIZE)



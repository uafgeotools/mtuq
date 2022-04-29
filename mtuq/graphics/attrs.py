import os

from matplotlib import pyplot
from os.path import join

from mtuq.util import warn


def plot_time_shifts(*args, **kwargs):
    """ Creates "spider plots" showing how time shifts vary geographically

    Within the specified directory, a separate PNG figure will be created for 
    each given component. Any components not present in the data will be 
    skipped.

    """
    kwargs.update({'attr_key': 'time_shift'})
    _plot_attrs(*args, **kwargs)


def plot_amplitude_ratios(*args, **kwargs):
    """ Creates "spider plots" showing how amplitude ratios vary geographically

    Within the specified directory, a separate PNG figure will be created for 
    each given component. Any components not present in the data will be 
    skipped.

    """
    kwargs.update({'attr_key': 'amplitude_ratio'})
    _plot_attrs(*args, **kwargs)


def plot_log_amplitude_ratios(*args, **kwargs):
    """ Creates "spider plots" showing how ln(Aobs/Asyn) varies geographically

    Within the specified directory, a separate PNG figure will be created for 
    each given component. Any components not present in the data will be 
    skipped.

    """
    kwargs.update({'attr_key': 'log_amplitude_ratio'})
    _plot_attrs(*args, **kwargs)



def _plot_attrs(dirname, attrs, stations, origin, source,
     attr_key='time_shift', components=['Z', 'R', 'T'], format='png', 
     _backend=_plot_matplotlib):

    if backend.lower()=='gmt':
        raise NotImplementedError

    os.makedirs(dirname, exist_ok=True)

    for component in components:
        attr_list = []
        station_list = []

        for _i, station in enumerate(stations):
            if component not in attrs[_i]:
                continue

            attr_list += [attrs[_i][component][attr_key]]
            station_list += [stations[_i]]

        if len(attr_list) > 0:
            filename = join(dirname, component+'.'+format)
            _backend(filename, attr_list, station_list, origin, source)


def _plot_matplotlib(filename, time_shifts, stations, origin, source, 
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


def _plot_pygmt(filename, time_shifts, stations, origin, source):
    raise NotImplementedError



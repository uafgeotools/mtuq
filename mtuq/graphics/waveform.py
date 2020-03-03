
#
# graphics/waveform.py - plots of observed and synthetic waveforms
#

import numpy as np
import matplotlib.pyplot as pyplot
import warnings

from matplotlib.font_manager import FontProperties
from mtuq.graphics.header import Header
from mtuq.util.signal import get_components
from obspy.geodetics import gps2dist_azimuth



#
# functions that generate entire figures
#

def plot_data_synthetics(filename, 
        data_bw,
        data_sw,
        synthetics_bw,
        synthetics_sw,
        stations,
        origin,
        header=None,
        total_misfit_bw=1., 
        total_misfit_sw=1., 
        normalize='maximum_amplitude',
        station_labels=True, 
        trace_labels=True):

    """ Creates CAP-style data/synthetics figure
    """
    if _isempty(data_bw):
        pass

    if _isempty(data_sw):
        raise Exception('Empty dataset')

    # how many stations have at least one trace?
    nstations = _count([data_bw, data_sw])


    # dimensions of subplot array
    nrows = nstations
    ncolumns = 6
    _irow = 0

    # column 0 - station labels
    # columns 1,2 - body wave traces
    # columns 3,4,5 - surface wave traces
    column_width_ratios = [0.1,0.4,0.4,1.,1.,1.]

    # figure dimensions in inches
    height = 1.4*nrows
    width = 15.
    margin_bottom = 0.25
    margin_top = 0.25
    margin_left = 0.25
    margin_right = 0.25

    if header:
        header_height = 1.75
        header_width = width
        header_margin_top = margin_top
        header_margin_left = margin_left
        height += header_height
    else:
        header_height = 0.
        header_width = 0.
        header_margin_top = 0.
        header_margin_bottom = 0.

    if station_labels:
        margin_left += 0.5

    height += margin_bottom
    height += margin_top
    width += margin_left
    width += margin_right

    # adjust dimensions if body waves absent
    if _isempty(data_bw):
        width -= 4
        column_width_ratios=[0.1,0.,0.,1.,1.,1.]
    

    # initialize figure
    fig, axes = pyplot.subplots(nrows, ncolumns, 
        figsize=(width, height),
        subplot_kw=dict(clip_on=False),
        gridspec_kw=dict(width_ratios=column_width_ratios)
        )

    pyplot.subplots_adjust(
        left=margin_left/width,
        right=1.-margin_right/width,
        bottom=margin_bottom/height,
        top=1.-(header_height+margin_top)/height,
        wspace=0.,
        hspace=0.,
        )

    _hide_axes(axes)


    # add labels above each column
    add_component_labels(axes,
        body_wave_labels=(not _isempty(data_bw)),
        surface_wave_labels=(not _isempty(data_sw)),
        )

    if header:
        # write CAPUAF-style header
        header.write(header_height, header_width,
            header_margin_left, header_margin_top)


    # determine maximum trace amplitudes
    max_amplitude_bw = 0.
    if data_bw.max() > max_amplitude_bw:
        max_amplitude_bw = data_bw.max()
    if synthetics_bw.max() > max_amplitude_bw:
        max_amplitude_bw = synthetics_bw.max()

    max_amplitude_sw = 0.
    if data_sw.max() > max_amplitude_sw:
        max_amplitude_sw = data_sw.max()
    if synthetics_sw.max() > max_amplitude_sw:
        max_amplitude_sw = synthetics_sw.max()


    #
    # loop over stations
    #

    for _i in range(len(stations)):

        # skip empty stations
        if len(data_bw[_i])==len(data_sw[_i])==0:
            continue

        # add station labels
        if station_labels:
            add_station_labels(axes[_irow, 0], stations[_i], origin)


        #
        # plot body wave traces
        #

        stream_dat = data_bw[_i]
        stream_syn = synthetics_bw[_i]

        for dat, syn in zip(stream_dat, stream_syn):
            component = dat.stats.channel[-1].upper()
            weight = getattr(dat, 'weight', 1.)

            # skip bad traces
            if component != syn.stats.channel[-1].upper():
                warnings.warn('Mismatched components, skipping...')
                continue
            elif weight==0.:
                continue

            # plot traces
            if component=='Z':
                axis = axes[_irow][1]
            elif component=='R':
                axis = axes[_irow][2]
            else:
                continue

            plot(axis, dat, syn)

            # normalize amplitudes
            if normalize=='trace_amplitude':
                max_trace = _max(dat, syn)
                ylim = [-2*max_trace, +2*max_trace]
                axis.set_ylim(*ylim)
            elif normalize=='maximum_amplitude':
                ylim = [-2*max_amplitude_bw, +2*max_amplitude_bw]
                axis.set_ylim(*ylim)

            if trace_labels:
                add_trace_labels(axis, dat, syn, total_misfit_bw)


        #
        # plot surface wave traces
        #

        stream_dat = data_sw[_i]
        stream_syn = synthetics_sw[_i]

        for dat, syn in zip(stream_dat, stream_syn):
            component = dat.stats.channel[-1].upper()
            weight = getattr(dat, 'weight', 1.)

            # skip bad traces
            if component != syn.stats.channel[-1].upper():
                warnings.warn('Mismatched components, skipping...')
                continue
            elif weight==0.:
                continue

            # plot traces
            if component=='Z':
                axis = axes[_irow][3]
            elif component=='R':
                axis = axes[_irow][4]
            elif component=='T':
                axis = axes[_irow][5]
            else:
                continue

            plot(axis, dat, syn)

            # normalize amplitude
            if normalize=='trace_amplitude':
                max_trace = _max(dat, syn)
                ylim = [-max_trace, +max_trace]
                axis.set_ylim(*ylim)
            elif normalize=='maximum_amplitude':
                ylim = [-max_amplitude_sw, +max_amplitude_sw]
                axis.set_ylim(*ylim)

            if trace_labels:
                add_trace_labels(axis, dat, syn, total_misfit_sw)

        _irow += 1

    pyplot.savefig(filename)
    pyplot.close()


def plot_data_greens(filename, 
        data_bw,
        data_sw,
        greens_bw,
        greens_sw,
        process_bw,
        process_sw,
        misfit_bw,
        misfit_sw,
        stations,
        origin,
        source,
        lune_dict,
        **kwargs):

    """ Creates CAP-style data/synthetics figure

    Similar to plot_data_synthetics, except provides different input argument
    syntax
    """
    event_name = filename.split('.')[0]
    model = _get_tag(greens_bw[0].tags, 'model')
    solver = _get_tag(greens_bw[0].tags, 'solver')

    greens_bw = greens_bw.select(origin)
    greens_sw = greens_sw.select(origin)
    _set_components(data_bw, greens_bw)
    _set_components(data_sw, greens_sw)

    synthetics_bw = greens_bw.get_synthetics(source)
    synthetics_sw = greens_sw.get_synthetics(source)

    # besides calculating misfit, these commands also set the trace attributes
    # used to align data and synthetics in the waveform plots
    total_misfit_bw = misfit_bw(data_bw, greens_bw, source, set_attributes=True)
    total_misfit_sw = misfit_sw(data_sw, greens_sw, source, set_attributes=True)

    if 'header' in kwargs:
        header = kwargs.pop('header')

    else:
        header = Header(event_name,
            process_bw, process_sw, misfit_bw, misfit_bw,
            model, solver, source, lune_dict, origin,
            total_misfit_bw, total_misfit_sw)

    plot_data_synthetics(filename,
        data_bw, data_sw, synthetics_bw, synthetics_sw, stations, origin,
        total_misfit_bw=total_misfit_bw, total_misfit_sw=total_misfit_sw,
        header=header, **kwargs)



#
# functions that act on individual axes
#

def plot(axis, dat, syn, label=None):
    """ Plots data and synthetics time series on current axes
    """
    t1,t2,nt,dt = _time_stats(dat)

    start = getattr(syn, 'start', 0)
    stop = getattr(syn, 'stop', len(syn.data))

    t = np.linspace(0,t2-t1,nt,dt)
    d = dat.data
    s = syn.data

    # ``clip_on=False`` and ``zorder=10`` should prevent the plotted data from
    # getting "clipped", but this doesn't seem to be happening in practice
    axis.plot(t, d, 'k', linewidth=1.5,
        clip_on=False, zorder=10)
    axis.plot(t, s[start:stop], 'r', linewidth=1.25, 
        clip_on=False, zorder=10)


def add_component_labels(axes, body_wave_labels=True, surface_wave_labels=True):
    """ Displays component name above each column
    """
    font = FontProperties()
    font.set_weight('bold')

    if body_wave_labels:
        ax = axes[0][1]
        pyplot.text(0.,0.70, 'Z', fontproperties=font, fontsize=16, 
            transform=ax.transAxes)

        ax = axes[0][2]
        pyplot.text(0.,0.70, 'R', fontproperties=font, fontsize=16, 
            transform=ax.transAxes)

    if surface_wave_labels:
        ax = axes[0][3]
        pyplot.text(0.,0.70, 'Z', fontproperties=font, fontsize=16,
            transform=ax.transAxes)

        ax = axes[0][4]
        pyplot.text(0.,0.70, 'R', fontproperties=font, fontsize=16,
            transform=ax.transAxes)

        ax = axes[0][5]
        pyplot.text(0.,0.70, 'T', fontproperties=font, fontsize=16,
            transform=ax.transAxes)


def add_station_labels(ax, station, origin):
    """ Displays station id, distance, and azimuth to the left of current axes
    """
    distance_in_m, azimuth, _ = gps2dist_azimuth(
        origin.latitude,
        origin.longitude,
        station.latitude,
        station.longitude)

    # display station name
    label = '.'.join([station.network, station.station])
    pyplot.text(-1.,0.50, label, fontsize=12, transform=ax.transAxes)

    # display distance
    distance = '%d km' % round(distance_in_m/1000.)
    pyplot.text(-1.,0.35, distance, fontsize=12, transform=ax.transAxes)

    # display azimuth
    azimuth =  '%d%s' % (round(azimuth), u'\N{DEGREE SIGN}')
    pyplot.text(-1.,0.20, azimuth, fontsize=12, transform=ax.transAxes)



def add_trace_labels(axis, dat, syn, total_misfit=1.):
    """ Adds CAP-style annotations to current axes
    """
    ymin = axis.get_ylim()[0]

    s = syn.data
    d = dat.data

    # display cross-correlation time shift
    time_shift = 0.
    time_shift += getattr(syn, 'time_shift', np.nan)
    time_shift += getattr(dat, 'static_time_shift', 0)
    axis.text(0.,(1/4.)*ymin, '%.2f' %time_shift, fontsize=12)

    # display maximum cross-correlation coefficient
    Ns = np.dot(s,s)**0.5
    Nd = np.dot(d,d)**0.5
    if Ns*Nd > 0.:
        max_cc = np.correlate(s, d, 'valid').max()
        max_cc /= (Ns*Nd)
        axis.text(0.,(2/4.)*ymin, '%.2f' %max_cc, fontsize=12)
    else:
        max_cc = np.nan
        axis.text(0.,(2/4.)*ymin, '%.2f' %max_cc, fontsize=12)

    # display percent of total misfit
    misfit = getattr(syn, 'misfit', np.nan)
    misfit /= total_misfit
    if misfit >= 0.1:
        axis.text(0.,(3/4.)*ymin, '%.1f' %(100.*misfit), fontsize=12)
    else:
        axis.text(0.,(3/4.)*ymin, '%.2f' %(100.*misfit), fontsize=12)


def _set_components(data, greens):
    for _i, stream in enumerate(data):
        components = get_components(stream)
        greens[_i]._set_components(components)


#
# utility functions
#

def _time_stats(trace):
    # returns time scheme
    return (
        float(trace.stats.starttime),
        float(trace.stats.endtime),
        trace.stats.npts,
        trace.stats.delta,
        )


def _count(datasets):
    # counts number of nonempty streams in dataset(s)
    count = 0
    for streams in zip(*datasets):
        for stream in streams:
            if len(stream) > 0:
                count += 1
                break
    return count


def _isempty(dataset):
    return bool(_count([dataset])==0)


def _max(dat, syn):
    # maximum amplitude of two traces
    return max(
        abs(dat.max()),
        abs(syn.max()))


def _hide_axes(axes):
    # hides axes lines, ticks, and labels
    for row in axes:
        for col in row:
            col.spines['top'].set_visible(False)
            col.spines['right'].set_visible(False)
            col.spines['bottom'].set_visible(False)
            col.spines['left'].set_visible(False)
            col.get_xaxis().set_ticks([])
            col.get_yaxis().set_ticks([])


def get_column_widths(data_bw, data_sw, width=1.):
    # creates argument used by pyplot.subplot

    for _i, stream in enumerate(data_bw):
        if len(stream) > 0:
            break
    for _j, stream in enumerate(data_sw):
        if len(stream) > 0:
            break

    stats_bw = data_bw[_i][0].stats
    stats_sw = data_sw[_j][0].stats
    len_bw = stats_bw.endtime-stats_bw.starttime
    len_sw = stats_sw.endtime-stats_sw.starttime

    width *= (2*len_bw+3*len_sw)**-1
    len_bw *= width
    len_sw *= width

    return\
        [len_bw, len_bw, len_sw, len_sw, len_sw]


def _get_tag(tags, pattern):
    for tag in tags:
        parts = tag.split(':')
        if parts[0]==pattern:
            return parts[1]
    else:
        return None


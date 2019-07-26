
import numpy as np
import matplotlib.pyplot as pyplot
import warnings
from mtuq.graphics.header import attach_header, generate_header


def plot_data_synthetics(filename, 
        data_bw, 
        data_sw, 
        synthetics_bw, 
        synthetics_sw, 
        total_misfit_bw=1., 
        total_misfit_sw=1., 
        normalize='maximum_amplitude',
        mt=None,
        title=None, 
        header=None,
        station_labels=True, 
        trace_labels=True):

    """ Creates CAP-style data/synthetics figure
    """

    # gather station metadata
    stations = data_bw.get_stations()
    assert stations == data_sw.get_stations()


    # keep track of maximum amplitudes
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
    # initialize figure
    #

    # dimensions of subplot array
    nrow = _count_nonempty([data_bw, data_sw]) # number of nonempty stations
    ncol = 5
    irow = 0


    # figure dimensions in inches
    height = 1.4*nrow
    width = 16.

    margin_bottom = 0.25
    margin_top = 0.25
    margin_left = 0.25
    margin_right = 0.25

    if station_labels:
        margin_left += 0.75

    height += margin_bottom
    height += margin_top
    width += margin_left
    width += margin_right


    # optional CAP-style header
    if not title:
        event_name = filename.split('.')[0]
        title = event_name

    if header:
        header_height = 2.5
        height += header_height
        fig = pyplot.figure(figsize=(width, height))
        attach_header(title, header, mt, header_height)

    else:
        header_height = 0.
        fig = pyplot.figure(figsize=(width, height))


    # adjust subplot spacing
    pyplot.subplots_adjust(
        left=margin_left/width,
        right=1.-margin_right/width,
        bottom=margin_bottom/height,
        top=1.-(margin_top+header_height)/height,
        wspace=0.,
        hspace=0.,
        )


    #
    # loop over stations
    #

    for _i in range(len(stations)):

        # skip empty stations
        if len(data_bw[_i])==len(data_sw[_i])==0:
            continue

        # add station labels
        if station_labels:
            meta = stations[_i]
            pyplot.subplot(nrow, ncol, ncol*irow+1)
            add_station_labels(meta)


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
                pyplot.subplot(nrow, ncol, ncol*irow+1)
                plot(dat, syn)
            elif component=='R':
                pyplot.subplot(nrow, ncol, ncol*irow+2)
                plot(dat, syn)
            else:
                continue

            # normalize amplitudes
            if normalize=='trace_amplitude':
                max_trace = _max(dat, syn)
                ylim = [-2*max_trace, +2*max_trace]
                pyplot.ylim(*ylim)
            elif normalize=='maximum_amplitude':
                ylim = [-2*max_amplitude_bw, +2*max_amplitude_bw]
                pyplot.ylim(*ylim)

            if trace_labels:
                add_trace_labels(dat, syn, total_misfit_bw)


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
                pyplot.subplot(nrow, ncol, ncol*irow+3)
                plot(dat, syn)
            elif component=='R':
                pyplot.subplot(nrow, ncol, ncol*irow+4)
                plot(dat, syn)
            elif component=='T':
                pyplot.subplot(nrow, ncol, ncol*irow+5)
                plot(dat, syn)
            else:
                continue

            # amplitude normalization
            if normalize=='trace_amplitude':
                max_trace = _max(dat, syn)
                ylim = [-max_trace, +max_trace]
                pyplot.ylim(*ylim)
            elif normalize=='maximum_amplitude':
                ylim = [-max_amplitude_sw, +max_amplitude_sw]
                pyplot.ylim(*ylim)

            if trace_labels:
                add_trace_labels(dat, syn, total_misfit_sw)

        irow += 1

    pyplot.savefig(filename)



def plot_data_greens(filename, 
        data, 
        greens,  
        process_data, 
        misfit, 
        source,
        origin,
        **kwargs):

    """ Creates CAP-style data/synthetics figure

    Similar to plot_data_synthetics, except provides different input argument
    syntax
    """
    event_name = filename.split('.')[0]

    # generate synthetics
    greens[0] = greens[0].select(origin)
    greens[1] = greens[1].select(origin)
    synthetics = []
    synthetics += [greens[0].get_synthetics(source)]
    synthetics += [greens[1].get_synthetics(source)]

    # evaluate misfit
    total_misfit = []
    total_misfit += [misfit[0](data[0], greens[0], source)]
    total_misfit += [misfit[1](data[1], greens[1], source)]

    header = generate_header(event_name,
        process_data[0], process_data[1], misfit[0], misfit[1],
        greens[0][0].model, 'syngine', source, origin.depth_in_m)

    plot_data_synthetics(filename, 
            data[0], data[1],
            synthetics[0], synthetics[1], 
            total_misfit[0], total_misfit[1],
            mt=source,
            header=header,
            **kwargs)



def plot(dat, syn, label=None):
    """ Plots data and synthetics time series on current axes
    """
    t1,t2,nt,dt = _time_stats(dat)

    start = getattr(syn, 'start', 0)
    stop = getattr(syn, 'stop', len(syn.data))

    meta = dat.stats
    d = dat.data
    s = syn.data

    ax = pyplot.gca()

    t = np.linspace(0,t2-t1,nt,dt)
    ax.plot(t, d, 'k')
    ax.plot(t, s[start:stop], 'r')

    _hide_axes(ax)


def add_station_labels(meta):
    """ Displays station id, distance, and azimuth to the left of current axes
    """
    ax = pyplot.gca()

    # display station name
    label = '.'.join([meta.network, meta.station])
    pyplot.text(-0.25,0.45, label, fontsize=12, transform=ax.transAxes)

    # display distance
    distance = '%d km' % round(meta.preliminary_distance_in_m/1000.)
    pyplot.text(-0.25,0.30, distance, fontsize=12, transform=ax.transAxes)

    # display azimuth
    azimuth =  '%d%s' % (round(meta.preliminary_azimuth), u'\N{DEGREE SIGN}')
    pyplot.text(-0.25,0.15, azimuth, fontsize=12, transform=ax.transAxes)

    _hide_axes(ax)



def add_trace_labels(dat, syn, total_misfit=1.):
    """ Adds CAP-style annotations to current axes
    """
    ax = pyplot.gca()
    ymin = ax.get_ylim()[0]

    s = syn.data
    d = dat.data

    # display cross-correlation time shift
    time_shift = getattr(syn, 'time_shift', np.nan)
    pyplot.text(0.,(1/4.)*ymin, '%.2f' %time_shift, fontsize=12)

    # display maximum cross-correlation coefficient
    Ns = np.dot(s,s)**0.5
    Nd = np.dot(d,d)**0.5
    if Ns*Nd > 0.:
        max_cc = np.correlate(s, d, 'valid').max()
        max_cc /= (Ns*Nd)
        pyplot.text(0.,(2/4.)*ymin, '%.2f' %max_cc, fontsize=12)
    else:
        max_cc = np.nan
        pyplot.text(0.,(2/4.)*ymin, '%.2f' %max_cc, fontsize=12)

    # display percent of total misfit
    misfit = getattr(syn, 'misfit', np.nan)
    misfit /= total_misfit
    if misfit >= 0.1:
        pyplot.text(0.,(3/4.)*ymin, '%.1f' %(100.*misfit), fontsize=12)
    else:
        pyplot.text(0.,(3/4.)*ymin, '%.2f' %(100.*misfit), fontsize=12)




### utilities


def _time_stats(trace):
    # returns time scheme
    return (
        float(trace.stats.starttime),
        float(trace.stats.endtime),
        trace.stats.npts,
        trace.stats.delta,
        )


def _count_nonempty(datasets):
    # counts number of nonempty streams in dataset(s)
    count = 0
    for streams in zip(*datasets):
        for stream in streams:
            if len(stream) > 0:
                count += 1
                break
    return count


def _max(dat, syn):
    # maximum amplitude of two traces
    return max(
        abs(dat.max()),
        abs(syn.max()))


def _hide_axes(ax):
    # hides axes lines, ticks, and labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])



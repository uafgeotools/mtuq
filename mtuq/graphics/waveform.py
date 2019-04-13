
import numpy as np
import matplotlib.pyplot as pyplot
import warnings


def plot_data_greens_mt(filename, data, greens, misfit, mt, **kwargs):
    """ Creates CAP-style data/synthetics figure

    Similar to plot_data_synthetics, except provides different input argument
    syntax
    """
    # generate synthetics
    greens[0].map(_set_components, data[0])
    greens[1].map(_set_components, data[1])
    synthetics = []
    synthetics += [greens[0].get_synthetics(mt)]
    synthetics += [greens[1].get_synthetics(mt)]

    # evaluate misfit
    total_misfit = []
    total_misfit += [misfit[0](data[0], greens[0], mt)]
    total_misfit += [misfit[1](data[1], greens[1], mt)]

    plot_data_synthetics(filename, data[0], data[1], 
        synthetics[0], synthetics[1], total_misfit[0], total_misfit[1],
        **kwargs)


def plot_data_synthetics(filename, data_bw, data_sw, 
        synthetics_bw, synthetics_sw, total_misfit_bw=1., total_misfit_sw=1.,
        annotate=False, normalize='maximum_amplitude'):
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


    # dimensions of subplot array
    nrow = count_nonempty([data_bw, data_sw])
    ncol = 6

    # initialize pyplot figure
    figsize = (16, 1.4*nrow)
    pyplot.figure(figsize=figsize)


    #
    # loop over stations
    #

    irow = 0
    for _i in range(len(stations)):

        # skip empty stations
        if len(data_bw[_i])==len(data_sw[_i])==0:
            continue

        # add station labels
        try:
            meta = stations[_i]
            pyplot.subplot(nrow, ncol, ncol*irow+1)
            station_labels(meta)
        except:
            meta = stream_dat[0].stats
            pyplot.subplot(nrow, ncol, ncol*irow+1)
            station_labels(meta)

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
                pyplot.subplot(nrow, ncol, ncol*irow+2)
                subplot(dat, syn)
            elif component=='R':
                pyplot.subplot(nrow, ncol, ncol*irow+3)
                subplot(dat, syn)
            else:
                continue

            # normalize amplitudes
            if normalize=='trace_by_trace':
                max_trace = _max(dat, syn)
                ylim = [-2*max_trace, +2*max_trace]
                pyplot.ylim(*ylim)
            elif normalize=='maximum_amplitude':
                ylim = [-2*max_amplitude_bw, +2*max_amplitude_bw]
                pyplot.ylim(*ylim)

            if annotate:
                time_shift = syn.time_shift
                misfit = syn.misfit
                misfit /= total_misfit_bw
                channel_labels(time_shift, misfit)


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
                pyplot.subplot(nrow, ncol, ncol*irow+4)
                subplot(dat, syn)
            elif component=='R':
                pyplot.subplot(nrow, ncol, ncol*irow+5)
                subplot(dat, syn)
            elif component=='T':
                pyplot.subplot(nrow, ncol, ncol*irow+6)
                subplot(dat, syn)
            else:
                continue

            # amplitude normalization
            if normalize=='trace_by_trace':
                max_trace = _max(dat, syn)
                ylim = [-max_trace, +max_trace]
                pyplot.ylim(*ylim)
            elif normalize=='maximum_amplitude':
                ylim = [-max_amplitude_sw, +max_amplitude_sw]
                pyplot.ylim(*ylim)

            if annotate:
                time_shift = syn.time_shift
                misfit = syn.misfit
                misfit /= total_misfit_bw
                channel_labels(time_shift, misfit)

        irow += 1

    pyplot.savefig(filename)



def subplot(dat, syn, label=None):
    t1,t2,nt,dt = time_stats(dat)

    start = getattr(syn, 'start', 0)
    stop = getattr(syn, 'stop', len(syn.data))

    meta = dat.stats
    d = dat.data
    s = syn.data

    ax = pyplot.gca()

    t = np.linspace(0,t2-t1,nt,dt)
    ax.plot(t, d, 'k')
    ax.plot(t, s[start:stop], 'r')

    hide_axes(ax)


def hide_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])



def station_labels(meta):
    ax = pyplot.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # display station name
    label = '.'.join([meta.network, meta.station])
    pyplot.text(0.6,0.5, label, fontsize=8)

    # display distance
    distance = '%d km' % round(meta.preliminary_distance_in_m/1000.)
    pyplot.text(0.6,0.3,distance, fontsize=8)

    # display azimuth
    azimuth =  '%d%s' % (round(meta.preliminary_azimuth), u'\N{DEGREE SIGN}')
    pyplot.text(0.6,0.1,azimuth, fontsize=8)


def channel_labels(dat, syn, ylim, total_misfit=1.):
    # CAP-style annotations
    time_shift = getattr(syn, 'time_shift', 'None')
    pyplot.text(0.,(1/4.)*ylim[0], '%.2f' %time_shift, fontsize=6)

    misfit = getattr(dat, 'misfit', 'None')
    sum_residuals /= total_misfit
    pyplot.text(0.,(2/4.)*ylim[0], '%.1e' %sum_residuals, fontsize=6)



### utilities


def time_stats(trace):
    # returns time scheme
    return (
        float(trace.stats.starttime),
        float(trace.stats.endtime),
        trace.stats.npts,
        trace.stats.delta,
        )


def _stack(*args):
    return np.column_stack(args)


def _set_components(greens, data):
    greens.components = [trace.stats.channel[-1] for trace in data]
    return greens


def count_nonempty(data):
    # counts number of stations with nonzero weights
    count = 0
    for streams in zip(*data):
        for stream in streams:
            if len(stream) > 0:
                count += 1
                continue
    return count

def _max(dat, syn):
    return max(
        abs(dat.max()),
        abs(syn.max()))


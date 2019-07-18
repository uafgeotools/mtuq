
import numpy as np
import matplotlib.pyplot as pyplot
import warnings
from matplotlib.font_manager import FontProperties


def plot_data_synthetics(filename, 
        data_bw, 
        data_sw, 
        synthetics_bw, 
        synthetics_sw, 
        process_bw=None,
        process_sw=None,
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

    if header==True:
        header = generate_header(event_name,
            process_bw, process_sw, misfit_bw, misfit_sw,
            model, 'syngine', best_mt, origins[0].depth_in_m)

    if header:
        header_height = 2.5
        height += header_height
        fig = pyplot.figure(figsize=(width, height))
        add_header(title, header, mt, header_height)

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



def plot_data_greens_mt(filename, 
        data, 
        greens,  
        process_data, 
        misfit, 
        mt,  
        **kwargs):

    """ Creates CAP-style data/synthetics figure

    Similar to plot_data_synthetics, except provides different input argument
    syntax
    """
    # generate synthetics
    synthetics = []
    synthetics += [greens[0].get_synthetics(mt)]
    synthetics += [greens[1].get_synthetics(mt)]

    # evaluate misfit
    total_misfit = []
    total_misfit += [misfit[0](data[0], greens[0], mt)]
    total_misfit += [misfit[1](data[1], greens[1], mt)]

    plot_data_synthetics(filename, 
            data[0], data[1],
            synthetics[0], synthetics[1], 
            process_data[0], process_data[1],
            total_misfit[0], total_misfit[1],
            mt=mt,
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


class Header(dict):
    """ A dictionary-like object containing header text

    Stores header text in a dictionary {position: text}, where position is an
    integer and text is a string. (This syntax is inspired by 
    matplotlib.pyplot.subplot.)

    .. example:

       Create a text header with one row and two columns:

           Header({1: 'text of column1', 2: 'text of column2'}, shape=[1,2])


    """
    def __init__(self, items, shape=np.array([])):
        super(Header, self).__init__(items)
        self.shape = shape


def add_header(title=None, header=None, mt=None, height=None):
    """ Adds header object to current figure
    """
    fig = pyplot.gcf()
    width, figure_height = fig.get_size_inches()

    x0 = 0.
    y0 = 1.-height/figure_height
    ax = fig.add_axes([x0, y0, 1., height/figure_height])
    ax.set_xlim([0., width])
    ax.set_ylim([0., height])
    ax = pyplot.gca()

    if title:
        _bold(title, 0.15, 0.7, ax, fontsize=16)

    for _i, text in header.items():
        ix = _i % header.shape[1]
        iy = _i / header.shape[1]
        px = float(ix) / header.shape[1]
        py = 1. - float(iy) / header.shape[0]
        px *= 0.4
        px += 0.15
        if title: 
            py -= 0.05
            py *= 0.5
        _text(text, px, py, ax, fontsize=14)

    # add beachball
    from obspy.imaging.beachball import beach
    beach = beach(mt, xy=(1., 1.), width=1.75, linewidth=0.5, facecolor='gray')
    ax = pyplot.gca()
    ax.add_collection(beach)

    _hide_axes(ax)


def generate_header(event_name, process_bw, process_sw, misfit_bw, misfit_sw,
    model, solver, mt, depth_in_m):
    """ Creates header object with CAP-style text
    """
    M0 = np.sqrt(0.5*np.sum(mt[0:3]**2.) + np.sum(mt[3:6]**2.))
    Mw = (np.log10(M0) - 9.1)/1.5

    norm_order = misfit_bw.norm_order
    assert norm_order==misfit_sw.norm_order
    norm = '$L%s$' % norm_order

    bw_T_min = process_bw.freq_max**-1
    bw_T_max = process_bw.freq_min**-1
    sw_T_min = process_sw.freq_max**-1
    sw_T_max = process_sw.freq_min**-1

    bw_win_len = process_bw.window_length
    sw_win_len = process_sw.window_length

    return Header(
        shape=np.array([4,4]),
        items={
            0: '$M_w$: %3.2f' % Mw,
            1: 'depth: %.1f km' % (depth_in_m/1000.),
            #2: 'CLVD: %.0f' % 0.,
            #3: 'ISO: %.0f' % 0.,
            4: 'model: %s' % model,
            5: 'solver: %s' % solver,
            #6: 'norm: %s' % norm,
            8: 'b.w. bandpass: %.1f - %.1f s' % (bw_T_min, bw_T_max),
            10: 's.w. bandpass: %.1f - %.1f s' % (sw_T_min, sw_T_max),
            12: 'b.w. window: %.1f s' % bw_win_len,
            14: 's.w. window: %.1f s' % sw_win_len})


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


def _text(text, x, y, ax, fontsize=12):
    pyplot.text(x, y, text, fontsize=fontsize, transform=ax.transAxes)


def _bold(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_weight('bold')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize, 
        transform=ax.transAxes)


def _italic(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_style('italic')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize,
        transform=ax.transAxes)



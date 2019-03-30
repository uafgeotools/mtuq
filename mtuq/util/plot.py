
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
import warnings
from obspy.imaging.beachball import beach, beachball



def plot_beachball(filename, mt):
    """ Plots source mechanism
    """
    beachball(mt, size=200, linewidth=2, facecolor='b')
    pyplot.savefig(filename)


def beachball_vs_depth(filename, mt_dict):
    n=len(mt_dict)

    fig = pyplot.figure(figsize=(n+1, 1))
    ax = pyplot.gca()

    depths = mt_dict.keys()
    mt_list = mt_dict.values()
    magnitudes =  [_magnitude(mt) for mt in mt_list]

    # create iterator
    zipped = zip(depths, mt_list, magnitudes)
    zipped = sorted(zipped, key=lambda x: x[0])

    # plot beachballs
    for _i, item in enumerate(zipped):
        depth_in_m, mt, magnitude = item

        # add beachball
        ax.add_collection(
            beach(mt, xy=(_i+1, 0), width=0.5))

        # add depth label
        label = str(depth_in_m/1000.)+' '+'km'
        x, y = _i+1, -0.5

        pyplot.text(x, y, label,
            fontsize=8,
            horizontalalignment='center')

        # add magnitude label
        label = '%2.1f' % magnitude
        x, y = _i+1, -0.33

        pyplot.text(x, y, label,
            fontsize=8,
            horizontalalignment='center')

    ax.set_aspect("equal")
    ax.set_xlim((0, n+1))
    ax.set_ylim((-0.5, +0.5))
    _invisible(ax)

    pyplot.savefig(filename)
    pyplot.close()


def misfit_vs_depth(filename, misfit_dict):
    fig = pyplot.figure()
    ax = pyplot.gca()

    pyplot.plot(misfit_dict.keys(), misfit_dict.values(), '.')

    pyplot.savefig(filename)
    pyplot.close()


def plot_data_greens_mt(filename, data_bw, data_sw, greens_bw, greens_sw, mt, 
        misfit_bw=None, misfit_sw=None, **kwargs):

    # generate synthetics
    synthetics_bw = greens_bw.get_synthetics(mt)
    synthetics_sw = greens_sw.get_synthetics(mt)

    # reevaluate misfit to get time shifts
    if misfit_bw:
        _ = misfit_bw(data_bw, greens_bw, mt)

    if misfit_sw:
        _ = misfit_sw(data_sw, greens_sw, mt)

    plot_data_synthetics(filename, data_bw, data_sw, 
        synthetics_bw, synthetics_sw, **kwargs)


def plot_data_synthetics(filename, data_bw_, data_sw_, synthetics_bw_, 
        synthetics_sw_, annotate=False, normalize=1):
    """ Creates CAP-style data/synthetics figure
    """

    # create figure object
    ncol = 6
    nrow = len(data_bw_)
    figsize = (16, 1.4*nrow)
    pyplot.figure(figsize=figsize)


    # determine axis limits
    max_bw = data_bw_.max()
    max_sw = data_sw_.max()

    irow = 0
    for data_bw, synthetics_bw, data_sw, synthetics_sw in zip(
        data_bw_, synthetics_bw_, data_sw_, synthetics_sw_):

        if len(data_bw)==len(data_sw)==0:
            continue
        try:
            id = data_bw.id
            meta = data_bw[0].stats
        except:
            id = data_sw.id
            meta = data_sw[0].stats

        pyplot.subplot(nrow, ncol, ncol*irow+1)

        # add station labels
        station_labels(meta)

        # plot body wave traces
        for dat, syn in zip(data_bw, synthetics_bw):
            component = dat.stats.channel[-1].upper()
            weight = getattr(dat, 'weight', 1.)

            if component != syn.stats.channel[-1].upper():
                warnings.warn('Mismatched components, skipping...')
                continue

            if weight==0.:
                continue

            # set axis limits
            if normalize==1:
                ymax = max_bw
                ylim = [-ymax, +ymax]
            elif normalize==2:
                _scale(dat)
                _scale(syn)
                ylim = [-1., +1.]
            else:
                ymax = _max(dat)
                ylim = [-ymax, +ymax]

            if component=='Z':
                pyplot.subplot(nrow, ncol, ncol*irow+2)
                subplot(dat, syn)

            elif component=='R':
                pyplot.subplot(nrow, ncol, ncol*irow+3)
                subplot(dat, syn)

            else:
                continue

            pyplot.ylim(*ylim)
            if annotate:
                channel_labels(dat, syn, ylim)


        # plot surface wave traces
        for dat, syn in zip(data_sw, synthetics_sw):
            component = dat.stats.channel[-1].upper()
            weight = getattr(dat, 'weight', 1.)

            if component != syn.stats.channel[-1].upper():
                warnings.warn('Mismatched components, skipping...')
                continue

            if weight==0.:
                continue

            # set axis limits
            if normalize==1:
                ymax = max_sw
                ylim = [-ymax, +ymax]
            elif normalize==2:
                _scale(dat)
                _scale(syn)
                ylim = [-1., +1.]
            else:
                ymax = _max(dat)
                ylim = [-ymax, +ymax]

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

            pyplot.ylim(*ylim)
            if annotate:
                channel_labels(dat, syn, ylim)

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

    _invisible(ax)


def _invisible(ax):
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
    pyplot.text(0.6,0.8, label, fontsize=12)

    try:
        # display distance and azimuth
        distance = '%d km' % round(meta.catalog_distance)
        azimuth =  '%d%s' % (round(meta.catalog_azimuth), u'\N{DEGREE SIGN}')
        pyplot.text(0.6,0.6,distance, fontsize=12)
        pyplot.text(0.6,0.4,azimuth, fontsize=12)
    except:
        pass


def channel_labels(dat, syn, ylim):
    # CAP-style annotations
    time_shift = getattr(syn, 'time_shift', 'None')
    pyplot.text(0.,(1/4.)*ylim[0], '%.2f' %time_shift, fontsize=6)

    sum_residuals = getattr(syn, 'sum_residuals', 'None')
    pyplot.text(0.,(2/4.)*ylim[0], '%.1e' %sum_residuals, fontsize=6)

    #label3 = getattr(syn, 'label3', 'None')
    #pyplot.text(0.,(3/4.)*ylim[0], '%.2f' %label3, fontsize=6)

    #label4 = getattr(syn, 'label4', 'None')
    #pyplot.text(0.,(4/4.)*ylim[0], '%.2f' %label4, fontsize=6)



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


def _scale(trace):
    dmax = max(abs(trace.data))
    if dmax > 0.: trace.data /= dmax


def _max(trace):
     return max(abs(trace.data))


def m_to_deg(distance_in_m):
    from obspy.geodetics import kilometers2degrees
    return kilometers2degrees(distance_in_m/1000., radius=6371.)

def km_to_deg(distance_in_m):
    from obspy.geodetics import kilometers2degrees
    return kilometers2degrees(distance_in_m, radius=6371.)


def _magnitude(mt):
    M = _asmatrix(mt)
    return (np.tensordot(M,M)/2.)*0.5


def _asmatrix(m):
    return np.array([
        [m[0], m[3], m[4]],
        [m[3], m[1], m[5]],
        [m[4], m[5], m[2]]])


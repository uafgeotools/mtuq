
import numpy as np
import matplotlib.pyplot as pyplot


def cap_plot(filename, data, synthetics, misfit):
    """ Creates cap-style plot
    """
    # reevaluate misfit to get time shifts
    for key in ['body_waves', 'surface_waves']:
        dat, syn, chi = data[key], synthetics[key], misfit[key]
        _ = chi(dat, syn)


    # how many rows, columns?
    nc = 6
    _, nr = shape(data)


    # create figure
    figsize = (16,1.4*nr)
    pyplot.figure(figsize=figsize)


    ir = 0
    for d1,s1,d2,s2 in zip(
        data['body_waves'], synthetics['body_waves'],
        data['surface_waves'], synthetics['surface_waves']):

        id = d1.id
        meta = d1.station

        # display station name
        pyplot.subplot(nr, nc, nc*ir+1)
        cap_station_labels(meta)

        # plot body waves
        for dat, syn in zip(d1, s1):
            component = dat.stats.channel[-1].upper()
            weight = dat.weight

            if not weight:
                continue

            if component=='Z':
                pyplot.subplot(nr, nc, nc*ir+2)
                cap_subplot(dat, syn)

            if component=='R':
                pyplot.subplot(nr, nc, nc*ir+3)
                cap_subplot(dat, syn)

        # plot surface waves
        for dat, syn in zip(d2, s2):
            component = dat.stats.channel[-1].upper()
            weight = dat.weight

            if not weight:
                continue

            if component=='Z':
                pyplot.subplot(nr, nc, nc*ir+4)
                cap_subplot(dat, syn)

            if component=='R':
                pyplot.subplot(nr, nc, nc*ir+5)
                cap_subplot(dat, syn)

            if component=='T':
                pyplot.subplot(nr, nc, nc*ir+6)
                cap_subplot(dat, syn)

        ir += 1

    pyplot.savefig(filename)


def cap_subplot(dat, syn, label=None, scale_type='normalize'):
    t1,t2,nt,dt = time_stats(dat)
    start = syn.start
    stop = syn.stop

    meta = dat.stats
    d = dat.data
    s = syn.data

    if scale_type=='default':
        s *= 100.
    elif scale_type=='normalize':
        d /= max(abs(d))
        s /= max(abs(s))

    t = np.linspace(0,t2-t1,nt,dt)
    pyplot.plot(t, d, t, s[start:stop])

    ax = pyplot.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    if label:
        pyplot.text(0.,0.6,meta.station, fontsize=10)



def cap_station_labels(meta):
    ax = pyplot.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # display station name
    label = '.'.join([meta.network, meta.station, meta.channels[0][:-1]])
    pyplot.text(0.6,0.8, label, fontsize=12)

    try:
        # display distance and azimuth
        distance = '%d km' % round(meta.catalog_distance)
        azimuth =  '%d%s' % (round(meta.catalog_azimuth), u'\N{DEGREE SIGN}')
        pyplot.text(0.6,0.6,distance, fontsize=12)
        pyplot.text(0.6,0.4,azimuth, fontsize=12)
    except:
        pass



def cap_channel_labels(meta):
    raise NotImplementedError




def time_stats(trace):
    if hasattr(trace, 'time_shift'):
        time_shift_npts = trace.time_shift

    return (
        float(trace.stats.starttime),
        float(trace.stats.endtime),
        trace.stats.npts,
        trace.stats.delta,
        )


def shape(dataset):
    nc = 0
    for i in dataset:
        nc += 1

    nr = 0
    for j in dataset[i]:
        nr += 1

    return nc, nr




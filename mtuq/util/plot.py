
import numpy as np
import matplotlib.pyplot as pyplot

from collections import defaultdict


def cap_plot(filename, data, synthetics):
    """ Creates cap-style plot
    """
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
        meta = data['body_waves'].get_station(id)

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


def cap_subplot(dat, syn, label=False):
    t1,t2,nt,dt = time_stats(dat)
    t = np.linspace(0,t2-t1,nt,dt)

    meta = dat.stats
    dat = dat.data
    syn = syn.data

    dat /= max(abs(dat))
    syn /= max(abs(syn))

    pyplot.plot(t,dat, 'k', t,syn,'r')
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
    pyplot.text(0.75,0.8,meta.station, fontsize=12)

    try:
        # display distance and azimuth
        distance = '%d km' % round(meta.catalog_distance)
        azimuth =  '%d deg' % round(meta.catalog_azimuth)
        pyplot.text(0.75,0.6,distance, fontsize=12)
        pyplot.text(0.75,0.4,azimuth, fontsize=12)
    except:
        pass



def cap_channel_labels(meta):
    raise NotImplementedError




def time_stats(trace):
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





import numpy as np
import matplotlib.pyplot as pyplot


def cap_plot(filename, data, greens, mt, weights):
    """ Creates cap-style plot
    """
    nc, nr = shape(data)
    ir = 0

    figsize = (8,1.4*nr)
    pyplot.figure(figsize=figsize)

    for d1,g1,d2,g2 in zip(
        data['body_waves'], greens['body_waves'],
        data['surface_waves'], greens['surface_waves']):

        s1 = g1.get_synthetics(mt)
        s2 = g2.get_synthetics(mt)

        pyplot.subplot(nr, nc, nc*ir+1)
        plot_dat_syn(d1[0], s1[0])

        pyplot.subplot(nr, nc, nc*ir+2)
        plot_dat_syn(d2[0], s2[0])

        ir += 1

    pyplot.savefig(filename)


def plot_dat_syn(dat, syn):
    t1,t2,nt,dt = time_stats(dat)
    t = np.linspace(0,t2-t1,nt,dt)

    metadata = dat.stats
    dat = dat.data
    syn = syn.data

    dat /= max(abs(dat))
    syn /= max(abs(syn))

    pyplot.plot(t,dat, 'k', t,syn,'r')
    pyplot.text(0.,0.6,metadata.station, fontsize=10)

    ax = pyplot.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


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


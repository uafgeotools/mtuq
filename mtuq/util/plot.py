
import numpy as np
from matplotlib import pyplot


def cap_plot(data, greens, mt):
    """ Creates cap-style plot
    """
    nc = len(data)
    nr = len(data.values()[0])

    pyplot.figure(figsize=(8,2*nr))

    for ir in range(nr):
        pyplot.subplot(nr, nc, nc*ir+1)
        dat = data['body_waves'][ir]
        syn = greens['body_waves'][ir].get_synthetics(mt)
        plot_dat_syn(dat[0], syn[0])

        pyplot.subplot(nr, nc, nc*ir+2)
        dat = data['surface_waves'][ir]
        syn = greens['surface_waves'][ir].get_synthetics(mt)
        plot_dat_syn(dat[0], syn[0])

    pyplot.savefig('tmp.png')
    import sys; sys.exit()


def plot_dat_syn(dat, syn):
    t1,t2,nt,dt = time_stats(dat)
    t = np.linspace(t1,t2,nt,dt)
    pyplot.plot(t,dat.data, 'k', t,syn.data,'r')


def time_stats(trace):
    return (
        float(trace.stats.starttime),
        float(trace.stats.endtime),
        trace.stats.npts,
        trace.stats.delta,
        )


import numpy as np
import matplotlib.pyplot as pyplot
import warnings
from obspy.imaging.beachball import beachball


def plot_waveforms(filename, data, synthetics, misfit=None, 
                   annotate=False, normalize=True):
    """ Creates cap-style data/synthetics comparison
    """

    # create figure object
    ncol = 6
    _, nrow = shape(data)
    figsize = (16, 1.4*nrow)
    pyplot.figure(figsize=figsize)


    # determine axis limits
    min_bw, max_bw = data['body_waves'].min(), data['body_waves'].max()
    min_sw, max_sw = data['surface_waves'].min(), data['surface_waves'].max()


    if misfit:
        # reevaluate misfit to get time shifts
        for key in ['body_waves', 'surface_waves']:
            dat, syn, func = data[key], synthetics[key], misfit[key]
            _ = func(dat, syn)

    irow = 0
    for data_bw, synthetics_bw, data_sw, synthetics_sw in zip(
        data['body_waves'], synthetics['body_waves'],
        data['surface_waves'], synthetics['surface_waves']):

        id = data_bw.id
        meta = data_bw.station

        # add station labels
        pyplot.subplot(nrow, ncol, ncol*irow+1)
        station_labels(meta)


        # plot body wave comparisons
        for dat, syn in zip(data_bw, synthetics_bw):
            component = dat.stats.channel[-1].upper()
            weight = getattr(dat, 'weight', 1.)

            if component != syn.stats.channel[-1].upper():
                warnings.warn('Skipping component')
                continue

            if weight==0.:
                continue

            if component=='Z':
                pyplot.subplot(nrow, ncol, ncol*irow+2)
                subplot(dat, syn)

            elif component=='R':
                pyplot.subplot(nrow, ncol, ncol*irow+3)
                subplot(dat, syn)

            else:
                continue

            if normalize:
                ylim = [min_bw, max_bw]
                pyplot.ylim(*ylim)
            else:
                ylim = [dat.data.min(), dat.data.max()]

            if annotate:
                # CAP-style annotations
                time_shift = getattr(syn, 'time_shift', 'None')
                pyplot.text(0.,(1/4.)*ylim[0], '%.2f' %time_shift, fontsize=6)

                sum_residuals = getattr(syn, 'sum_residuals', 'None')
                pyplot.text(0.,(2/4.)*ylim[0], '%.1e' %sum_residuals, fontsize=6)

                #label3 = getattr(syn, 'label3', 'None')
                #pyplot.text(0.,(3/4.)*ylim[0], '%.2f' %label3, fontsize=6)

                #label4 = getattr(syn, 'label4', 'None')
                #pyplot.text(0.,(4/4.)*ylim[0], '%.2f' %label4, fontsize=6)


        # plot surface wave comparisons
        for dat, syn in zip(data_sw, synthetics_sw):
            component = dat.stats.channel[-1].upper()
            weight = getattr(dat, 'weight', 1.)

            if component != syn.stats.channel[-1].upper():
                warnings.warn('Skipping component')
                continue

            if weight==0.:
                continue

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

            if normalize:
                ylim = [min_sw, max_sw]
                pyplot.ylim(*ylim)
            else:
                ylim = [dat.data.min(), dat.data.max()]

            if annotate:
                # CAP-style annotations
                time_shift = getattr(syn, 'time_shift', 'None')
                pyplot.text(0.,(1/4.)*ylim[0], '%.2f' %time_shift, fontsize=6)

                sum_residuals = getattr(syn, 'sum_residuals', 'None')
                pyplot.text(0.,(2/4.)*ylim[0], '%.1e' %sum_residuals, fontsize=6)

                #label3 = getattr(syn, 'label3', 'None')
                #pyplot.text(0.,(3/4.)*ylim[0], '%.2f' %label3, fontsize=6)

                #label4 = getattr(syn, 'label4', 'None')
                #pyplot.text(0.,(4/4.)*ylim[0], '%.2f' %label4, fontsize=6)

        irow += 1

    pyplot.savefig(filename)
    #pyplot.show()


def subplot(dat, syn, label=None, normalize=False):
    t1,t2,nt,dt = time_stats(dat)

    start = getattr(syn, 'start', 0)
    stop = getattr(syn, 'stop', len(syn.data))

    meta = dat.stats
    d = dat.data
    s = syn.data

    if normalize:
        d0 = max(abs(d))
        s0 = max(abs(s))
        if d0 > 0.: d /= d0
        if s0 > 0.: s /= s0

    t = np.linspace(0,t2-t1,nt,dt)
    pyplot.plot(t, d, t, s[start:stop])

    ax = pyplot.gca()
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



def channel_labels(meta):
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




def mesh2grid(v, x, z):
    """ Interpolates from an unstructured coordinates (mesh) to a structured 
        coordinates (grid)
    """
    lx = x.max() - x.min()
    lz = z.max() - z.min()
    nn = v.size
    mesh = _stack(x, z)

    nx = 100
    nz = 100
    dx = lx/nx
    dz = lz/nz

    # construct structured grid
    x = np.linspace(x.min(), x.max(), nx)
    z = np.linspace(z.min(), z.max(), nz)
    X, Z = np.meshgrid(x, z)
    grid = _stack(X.flatten(), Z.flatten())

    # interpolate to structured grid
    V = scipy.interpolate.griddata(mesh, v, grid, 'linear')

    # workaround edge issues
    if np.any(np.isnan(V)):
        W = scipy.interpolate.griddata(mesh, v, grid, 'nearest')
        for i in np.where(np.isnan(V)):
            V[i] = W[i]

    return X,Z,np.reshape(V, (nz, nx))


def _stack(*args):
    return np.column_stack(args)



def plot_beachball(filename, mt):
    beachball(mt, size=200, linewidth=2, facecolor='b')
    pyplot.savefig(filename)


